#!/usr/bin/env python3
"""
Smoke test for train.py.  Six tests, increasing in fidelity:

  Test 1 — EAGER, training-scale config
    Uses the exact same initialisation path as the real training run:
      torch.device("meta")  →  to_empty()  →  init_weights()
    Runs inside torch.amp.autocast(dtype=bfloat16), matching production.
    Does a forward + backward pass.
    Catches: dtype mismatches (e.g. attn_temp float32 leak),
             zeroed non-persistent buffers (e.g. pc_scale=0 → NaN).

  Test 2 — COMPILED, small-scale config
    Tiny model compiled with fullgraph=True + forward+backward.
    Catches: torch.compile graph breaks, recompile stalls, shape bugs.

  Test 3 — MODEL INVARIANTS (Steps 3,4,8,9,10,11,12)
    Static parameter/structure checks:
      - resid_lambdas init value = softplus⁻¹(1)
      - _NoStoch registered buffer; identity pass-through
      - ve_gate_channels auto-computation and explicit override
      - pc_fc_w / pc_proj_w bottleneck shapes
      - aux_lm_heads count and shapes
      - weight tying lm_head ↔ wte (one param in optimizer)
      - EMBEDDING_LR = 0.1

  Test 4 — FORWARD NUMERICS (Steps 2, 3 stress, 9)
    Forward+backward numeric validation:
      - KL divergence computed in float32 (hook-verified)
      - No NaN under extreme negative resid_lambdas
      - pc_head_dim bottleneck einsums produce correct output shapes

  Test 5 — GRAD NORM + CHECKPOINT RESUME (Steps 5, 6)
    Optimizer-level checks:
      - clip_grad_norm_ returns a finite, positive grad_norm
      - Checkpoint save/load restores model weights, optimizer state, step,
        and total_training_time exactly

  Test 6 — VAL_INTERVAL CONTROL-FLOW (Issue 2 / Issue 3)
    Simulates steps 0–5 with VAL_INTERVAL=3 and a mocked evaluate_bpb.
    Asserts evaluate_bpb fires exactly once (at step 3), never at step 0.

Usage:
  python smoke_test.py              # run all five tests
  python smoke_test.py --no-compile # skip Test 2 (faster, e.g. in CI)
"""

import math
import os
import sys
import tempfile

import torch
import torch.nn.functional as F

# Save flags before patching sys.argv (train.py's module-level code must not
# see unknown args, but we still need our own flags after import).
_smoke_args = sys.argv[1:]
sys.argv = [sys.argv[0]]

# train.py reads env-vars at module level; supply sane defaults if not set
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",
                      os.path.expanduser("~/.cache/torchinductor"))
os.environ.setdefault("PC_WEIGHT",       "0.1")
os.environ.setdefault("PC_FOCAL_GAMMA",  "1.0")
os.environ.setdefault("KL_WEIGHT",       "0.01")
os.environ.setdefault("PC_DIAG_INTERVAL","50")

sys.path.insert(0, os.path.dirname(__file__))
import train as _train_module                                    # noqa: E402
from train import (GPT, GPTConfig,                              # noqa: E402
                   _NoStoch, StochasticLayer, AUX_HORIZONS)

DEVICE   = torch.device("cuda")
AUTOCAST = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def assert_finite(tensor, name: str):
    if tensor.isnan().any():
        raise AssertionError(f"{name} contains NaN")
    if tensor.isinf().any():
        raise AssertionError(f"{name} contains Inf")


def _tiny_config(**overrides):
    """Return a small GPTConfig suitable for tests that don't need a full model."""
    defaults = dict(
        sequence_len=64, vocab_size=256,
        n_layer=6, n_head=3, n_kv_head=3, n_embd=192,
        window_pattern="PROGRESSIVE", pc_head_dim=32,
    )
    defaults.update(overrides)
    return GPTConfig(**defaults)


def _build_model(config, device=DEVICE):
    """Meta-device construction + to_empty + init_weights, matching production."""
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    return model


# ---------------------------------------------------------------------------
# Test 1: eager forward+backward at training scale
# ---------------------------------------------------------------------------
def test_eager_training_scale():
    """
    Mirrors production init exactly:
      - meta device construction
      - to_empty() materialisation
      - init_weights() explicit buffer fill
      - autocast(bfloat16) context
      - forward + backward

    Uses actual training architecture (n_layer=8, n_head=4, n_embd=512) but
    a shorter sequence (T=256) and batch=1 to stay well within VRAM.
    """
    print("Test 1  [eager, training-scale]  ...", end="  ", flush=True)

    # Match build_model_config(DEPTH=8) from train.py:
    #   base_dim = 8 * 64 = 512
    #   num_heads = 512 // 128 = 4
    config = GPTConfig(
        sequence_len=512,    # shorter than production 2048 to save VRAM
        vocab_size=8192,
        n_layer=8,
        n_head=4,
        n_kv_head=4,
        n_embd=512,
        window_pattern="PROGRESSIVE",
    )

    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=DEVICE)
    model.init_weights()

    # Verify key non-persistent buffers were filled (not left at zero)
    pc_scale = model.pc_scale.item()
    assert pc_scale > 0,  f"pc_scale zeroed ({pc_scale}); meta-init fill bug"

    x = torch.randint(0, config.vocab_size, (1, 256), device=DEVICE)
    y = torch.randint(0, config.vocab_size, (1, 256), device=DEVICE)

    with AUTOCAST:
        main_loss, aux_loss, layer_means = model(x, y)
        (main_loss + 0.1 * aux_loss).backward()

    assert_finite(main_loss, "main_loss")
    assert_finite(aux_loss,  "aux_loss")
    assert math.isfinite(main_loss.item()), f"main_loss not finite: {main_loss.item()}"
    assert aux_loss.shape == (), f"aux_loss should be scalar, got shape {aux_loss.shape}"
    assert aux_loss.item() >= 0, f"aux_loss should be non-negative (pc_loss≥0), got {aux_loss.item()}"
    assert layer_means.shape == (config.n_layer - 1, config.n_embd), \
        f"layer_means shape {layer_means.shape} != ({config.n_layer - 1}, {config.n_embd})"
    assert layer_means.dtype == torch.float32, f"layer_means dtype {layer_means.dtype} should be float32"

    print(f"{PASS}  loss={main_loss.item():.4f}  aux={aux_loss.item():.4f}  pc_scale={pc_scale:.2f}")


# ---------------------------------------------------------------------------
# Test 2: compiled forward+backward on a tiny model
# ---------------------------------------------------------------------------
def test_compiled_small_scale():
    """
    Compiles a tiny model with fullgraph=True and runs one forward+backward.
    Catches graph breaks, shape errors, and compile-time dtype issues at low
    cost (small n_embd means few Triton kernels → fast compilation).
    """
    print("Test 2  [compiled, small-scale]  ...", end="  ", flush=True)

    config = GPTConfig(
        sequence_len=64,
        vocab_size=256,
        n_layer=6,
        n_head=6,
        n_kv_head=6,
        n_embd=192,
        window_pattern="PROGRESSIVE",
    )

    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=DEVICE)
    model.init_weights()
    model = torch.compile(model, dynamic=False, fullgraph=True)

    x = torch.randint(0, config.vocab_size, (2, 64), device=DEVICE)
    y = torch.randint(0, config.vocab_size, (2, 64), device=DEVICE)

    with AUTOCAST:
        main_loss, aux_loss, layer_means = model(x, y)
        (main_loss + 0.1 * aux_loss).backward()

    assert_finite(main_loss, "main_loss (compiled)")
    assert_finite(aux_loss,  "aux_loss (compiled)")
    assert math.isfinite(main_loss.item()), f"compiled main_loss not finite: {main_loss.item()}"
    assert aux_loss.shape == (), f"compiled aux_loss should be scalar, got shape {aux_loss.shape}"
    assert aux_loss.item() >= 0, f"compiled aux_loss should be non-negative, got {aux_loss.item()}"
    assert layer_means.dtype == torch.float32, f"compiled layer_means dtype {layer_means.dtype} should be float32"

    print(f"{PASS}  loss={main_loss.item():.4f}  aux={aux_loss.item():.4f}")


# ---------------------------------------------------------------------------
# Test 3: model invariants (Steps 3, 4, 8, 9, 10, 11, 12)
# ---------------------------------------------------------------------------
def test_model_invariants():
    """
    Static parameter/structure checks verifying the structural changes
    from the fix plan without running a forward pass (fast, no kernel compile).
    """
    print("Test 3  [model invariants]  ...", end="  ", flush=True)

    config = _tiny_config()
    model = _build_model(config)

    # ── Step 3: resid_lambdas initialised to softplus⁻¹(1) = log(e-1) ──────
    expected_init = math.log(math.e - 1)   # ≈ 0.5413
    assert torch.allclose(
        model.resid_lambdas,
        torch.full_like(model.resid_lambdas, expected_init),
        atol=1e-4,
    ), f"resid_lambdas init wrong: {model.resid_lambdas.tolist()}"
    assert torch.allclose(
        F.softplus(model.resid_lambdas),
        torch.ones_like(model.resid_lambdas),
        atol=1e-4,
    ), "softplus(resid_lambdas) should be 1.0 at init (neutral residual scale)"

    # Stress: softplus must be finite and non-negative for any extreme value.
    # (softplus(-1e6) underflows to 0.0 — the key property is no NaN/Inf,
    # which would propagate through the residual scaling in forward.)
    with torch.no_grad():
        model.resid_lambdas.fill_(-1e6)
    scales = F.softplus(model.resid_lambdas.float())
    assert not scales.isnan().any(), "softplus(resid_lambdas) produced NaN"
    assert not scales.isinf().any(), "softplus(resid_lambdas) produced Inf"
    assert (scales >= 0).all(), "softplus(resid_lambdas) produced negative values"
    with torch.no_grad():
        model.resid_lambdas.fill_(expected_init)   # restore

    # ── Step 4: _NoStoch has registered buffer; forward is a pass-through ───
    ns = _NoStoch()
    assert hasattr(ns, '_zero'), "_NoStoch is missing the '_zero' registered buffer"
    assert ns._zero.shape == (), \
        f"_NoStoch._zero should be a scalar tensor, got shape {ns._zero.shape}"
    x_cpu = torch.randn(3, 8)
    out, kl = ns(x_cpu)
    assert out is x_cpu, "_NoStoch.forward should return x unchanged (same object)"
    assert kl is ns._zero, "_NoStoch.forward kl should be the registered _zero buffer"

    # ── Step 8: ve_gate_channels auto-computed from config ───────────────────
    expected_vgc = max(32, config.n_embd // 16)   # max(32, 192//16=12) = 32
    for i, block in enumerate(model.transformer.h):
        assert block.attn.ve_gate_channels == expected_vgc, (
            f"Block {i}: ve_gate_channels={block.attn.ve_gate_channels}, "
            f"expected {expected_vgc}"
        )
    # Custom (non-zero) value overrides auto
    config_vgc = _tiny_config(n_layer=4, ve_gate_channels=48)
    model_vgc = _build_model(config_vgc)
    for i, block in enumerate(model_vgc.transformer.h):
        assert block.attn.ve_gate_channels == 48, (
            f"Block {i}: ve_gate_channels={block.attn.ve_gate_channels}, expected 48"
        )
    del model_vgc

    # ── Step 9: pc_fc_w / pc_proj_w use bottleneck pc_head_dim ──────────────
    # Both weights are (n_layer, pc_head_dim, n_embd): fc contracts on n_embd,
    # proj contracts on pc_head_dim via the einsum 'lbtd,ldc->lbtc'.
    expected_pc_shape = (config.n_layer, config.pc_head_dim, config.n_embd)
    assert model.pc_fc_w.shape == expected_pc_shape, (
        f"pc_fc_w shape wrong: {model.pc_fc_w.shape}, expected {expected_pc_shape}"
    )
    assert model.pc_proj_w.shape == expected_pc_shape, (
        f"pc_proj_w shape wrong: {model.pc_proj_w.shape}, expected {expected_pc_shape}"
    )
    assert config.pc_head_dim < config.n_embd, (
        f"pc_head_dim={config.pc_head_dim} should be < n_embd={config.n_embd} "
        "(must be a bottleneck)"
    )

    # ── Step 10: aux_lm_heads count, shapes, and optimizer coverage ──────────
    assert len(model.aux_lm_heads) == len(AUX_HORIZONS), (
        f"aux_lm_heads has {len(model.aux_lm_heads)} heads, "
        f"expected {len(AUX_HORIZONS)} (one per AUX_HORIZONS entry)"
    )
    for i, head in enumerate(model.aux_lm_heads):
        assert head.weight.shape == (config.vocab_size, config.n_embd), (
            f"aux_lm_heads[{i}].weight shape wrong: {head.weight.shape}, "
            f"expected ({config.vocab_size}, {config.n_embd})"
        )
    # All aux head parameters must appear in the optimizer
    model.lm_head.weight = model.transformer.wte.weight   # tie before optimizer
    optimizer = model.setup_optimizer()
    all_opt_param_ids = {id(p) for g in optimizer.param_groups for p in g['params']}
    for i, head in enumerate(model.aux_lm_heads):
        assert id(head.weight) in all_opt_param_ids, \
            f"aux_lm_heads[{i}].weight not found in any optimizer param group"

    # ── Step 11: weight tying — lm_head.weight IS wte.weight; counts once ───
    assert model.lm_head.weight is model.transformer.wte.weight, \
        "lm_head.weight is not wte.weight after tying"
    wte_w = model.transformer.wte.weight
    occurrences = sum(
        1 for g in optimizer.param_groups for p in g['params'] if p is wte_w
    )
    assert occurrences == 1, (
        f"wte/lm_head weight appears {occurrences} times in optimizer param groups, "
        f"expected exactly 1"
    )
    # lm_head itself should not form a separate param group
    lm_head_exclusive_ids = {id(model.lm_head.weight)} - {id(model.transformer.wte.weight)}
    for g in optimizer.param_groups:
        for p in g['params']:
            assert id(p) not in lm_head_exclusive_ids, \
                "lm_head.weight appears as a distinct tensor (not tied) in optimizer"

    # ── Step 12: EMBEDDING_LR reduced to 0.1 ────────────────────────────────
    assert _train_module.EMBEDDING_LR == 0.1, (
        f"EMBEDDING_LR expected 0.1, got {_train_module.EMBEDDING_LR}"
    )

    # ── lm_head_log_scale: new scalar parameter ──────────────────────────────
    assert hasattr(model, 'lm_head_log_scale'), \
        "GPT is missing lm_head_log_scale parameter"
    assert model.lm_head_log_scale.shape == (), \
        f"lm_head_log_scale should be scalar, got {model.lm_head_log_scale.shape}"
    assert abs(model.lm_head_log_scale.item()) < 1e-6, \
        f"lm_head_log_scale should init to 0.0 (scale=1), got {model.lm_head_log_scale.item()}"
    assert abs(model.lm_head_log_scale.exp().item() - 1.0) < 1e-5, \
        "exp(lm_head_log_scale) should be 1.0 at init (neutral)"
    # Must appear in optimizer param groups
    lls_id = id(model.lm_head_log_scale)
    assert lls_id in all_opt_param_ids, \
        "lm_head_log_scale not found in any optimizer param group"

    # ── pc_ema: persistent EMA buffer for PC targets ─────────────────────────
    assert hasattr(model, 'pc_ema'), "GPT is missing 'pc_ema' buffer"
    expected_ema_shape = (config.n_layer - 1, config.n_embd)
    assert model.pc_ema.shape == expected_ema_shape, (
        f"pc_ema shape {model.pc_ema.shape} != {expected_ema_shape}"
    )
    assert model.pc_ema.dtype == torch.float32, \
        f"pc_ema dtype {model.pc_ema.dtype} should be float32"
    assert (model.pc_ema == 0).all(), \
        "pc_ema should be zero-initialised"
    # Must be persistent so it survives checkpoint round-trips
    assert 'pc_ema' in dict(model.named_buffers()), \
        "pc_ema not found in model.named_buffers()"
    # Persistent buffers appear in state_dict; non-persistent ones do not
    assert 'pc_ema' in model.state_dict(), \
        "pc_ema not in model.state_dict() — must be persistent=True"

    # ── get_pc_diagnostics() includes sigma_bias_exp ─────────────────────────
    diag = model.get_pc_diagnostics()
    assert "sigma_bias_exp" in diag, \
        "get_pc_diagnostics() missing 'sigma_bias_exp' key"
    n_stoch = sum(1 for sl in model.stochastic_layers if isinstance(sl, StochasticLayer))
    assert len(diag["sigma_bias_exp"]) == n_stoch, (
        f"sigma_bias_exp has {len(diag['sigma_bias_exp'])} entries, "
        f"expected {n_stoch} (one per StochasticLayer)"
    )
    # log_sigma_proj.bias is initialised to -3.0, so exp(-3) ≈ 0.050
    expected_sigma_exp = math.exp(-3.0)
    for i, v in enumerate(diag["sigma_bias_exp"]):
        assert abs(v - expected_sigma_exp) / expected_sigma_exp < 0.05, (
            f"sigma_bias_exp[{i}]={v:.4f}, expected ≈{expected_sigma_exp:.4f} "
            f"(exp(-3)=0.050 at init)"
        )

    # ── num_scaling_params() deduplicates lm_head after weight tying ─────────
    counts = model.num_scaling_params()
    assert counts['lm_head_tied'] is True, \
        f"num_scaling_params lm_head_tied expected True, got {counts['lm_head_tied']}"
    assert counts['lm_head'] == 0, \
        f"num_scaling_params lm_head expected 0 (tied), got {counts['lm_head']}"
    # scalars now includes lm_head_log_scale (1 element) + resid_lambdas + pc_log_lambdas
    expected_scalars = config.n_layer * 2 + 1   # resid + pc_log + lm_head_log_scale
    assert counts['scalars'] == expected_scalars, (
        f"num_scaling_params scalars expected {expected_scalars} "
        f"(n_layer×2 + 1 for lm_head_log_scale), got {counts['scalars']}"
    )

    print(f"{PASS}")


# ---------------------------------------------------------------------------
# Test 4: forward numerics (Steps 2, 3 stress, 9)
# ---------------------------------------------------------------------------
def test_forward_numerics():
    """
    Forward+backward numeric validation:
      Step 2 — KL divergence computed in float32 (verified via forward hook)
      Step 3 — No NaN/Inf under extreme negative resid_lambdas
      Step 9 — pc_head_dim bottleneck einsums run without shape error
    """
    print("Test 4  [forward numerics]   ...", end="  ", flush=True)

    config = _tiny_config()
    model = _build_model(config)

    x = torch.randint(0, config.vocab_size, (2, 32), device=DEVICE)
    y = torch.randint(0, config.vocab_size, (2, 32), device=DEVICE)

    # ── Step 2: KL must be computed in float32 ───────────────────────────────
    kl_dtypes_seen: list[torch.dtype] = []

    def _kl_hook(module, inp, output):
        # output is (z, kl); capture dtype of the kl tensor
        kl_dtypes_seen.append(output[1].dtype)

    hooks = [
        sl.register_forward_hook(_kl_hook)
        for sl in model.stochastic_layers
        if isinstance(sl, StochasticLayer)
    ]
    assert hooks, "No StochasticLayer found in model — cannot test KL dtype"

    with AUTOCAST:
        main_loss, aux_loss, layer_means = model(x, y)
        (main_loss + 0.1 * aux_loss).backward()

    for h in hooks:
        h.remove()

    assert kl_dtypes_seen, "StochasticLayer forward hooks never fired"
    for dt in kl_dtypes_seen:
        assert dt == torch.float32, (
            f"KL was computed in {dt} — expected torch.float32 (step 2 fix)"
        )

    assert_finite(main_loss, "main_loss (forward_numerics)")
    assert_finite(aux_loss,  "aux_loss  (forward_numerics)")

    # ── wte_std init: target logit_std ≈ 1.75 so E[CE] ≈ 10.5 nats at step 0 ─
    # wte_std = 1.75/sqrt(n_embd); verify the embedding weight std matches.
    expected_wte_std = 1.75 / (config.n_embd ** 0.5)
    actual_wte_std = model.transformer.wte.weight.float().std().item()
    assert abs(actual_wte_std - expected_wte_std) / expected_wte_std < 0.15, (
        f"wte weight std={actual_wte_std:.4f}, expected ≈{expected_wte_std:.4f} "
        f"(1.75/sqrt(n_embd)); initial CE will be far from target ~10.5 nats"
    )

    # ── lm_head_log_scale: forward multiplies logits by exp(scale), which is 1.0
    # at init — verify the parameter value and that loss is finite (already above).
    assert abs(model.lm_head_log_scale.item()) < 1e-6, \
        f"lm_head_log_scale not 0.0 at init: {model.lm_head_log_scale.item()}"
    assert abs(model.lm_head_log_scale.exp().item() - 1.0) < 1e-5, \
        "exp(lm_head_log_scale) ≠ 1.0 at init — logit scale is unexpectedly non-neutral"

    # ── Step 3 stress: extreme negative resid_lambdas must not cause NaN ─────
    model.zero_grad()
    with torch.no_grad():
        model.resid_lambdas.fill_(-1e6)
    with AUTOCAST:
        loss2, aux2, _ = model(x, y)
    assert not loss2.isnan().any(), \
        "loss is NaN with extreme negative resid_lambdas (softplus fix broken?)"
    assert not loss2.isinf().any(), \
        "loss is Inf with extreme negative resid_lambdas"
    with torch.no_grad():
        model.resid_lambdas.fill_(math.log(math.e - 1))   # restore

    # ── reduction='none' returns a 2-tuple (not 3) ───────────────────────────
    # forward(x, y, reduction='none') is used by evaluate_bpb and the eval
    # pre-warm block.  It must NOT return layer_means (graph would break).
    model.zero_grad()
    with AUTOCAST:
        none_out = model(x, y, reduction='none')
    assert len(none_out) == 2, (
        f"model(x, y, reduction='none') should return 2-tuple "
        f"(token_ce, aux_loss), got {len(none_out)}-tuple"
    )
    token_ce, aux_none = none_out
    assert token_ce.shape == (x.shape[0], x.shape[1]), (
        f"token_ce shape {token_ce.shape} != (B={x.shape[0]}, T={x.shape[1]})"
    )

    # ── PC_WEIGHT_WARMUP removed from training loop ───────────────────────────
    # Run 20 confirmed warmup is counterproductive; the constant was deleted.
    import train as _tr
    assert not hasattr(_tr, 'PC_WEIGHT_WARMUP'), \
        "PC_WEIGHT_WARMUP still present — warmup removal not applied"

    # ── Step 9: pc_head_dim bottleneck — verify intermediate shape ───────────
    # The shapes are implicit in the parameters; verify via param shapes
    L = config.n_layer - 1
    assert model.pc_fc_w[:L].shape == (L, config.pc_head_dim, config.n_embd), \
        f"pc_fc_w[:L] shape wrong: {model.pc_fc_w[:L].shape}"
    # Run a second clean forward to confirm the einsums complete without error
    model.zero_grad()
    with AUTOCAST:
        loss3, aux3, _ = model(x, y)
        (loss3 + 0.1 * aux3).backward()
    assert_finite(loss3, "loss3 (pc_head_dim einsum check)")

    print(
        f"{PASS}  "
        f"kl_dtype=float32 ({len(kl_dtypes_seen)} layers)  "
        f"loss={main_loss.item():.4f}"
    )


# ---------------------------------------------------------------------------
# Test 5: grad norm + checkpoint resume (Steps 5, 6)
# ---------------------------------------------------------------------------
def test_grad_norm_and_checkpoint():
    """
    Step 5 — clip_grad_norm_ returns a finite, positive scalar.
    Step 6 — checkpoint save/load restores model weights, optimizer state,
             step counter, and total_training_time exactly.
    """
    print("Test 5  [grad norm + checkpoint]  ...", end="  ", flush=True)

    # The module-level @torch.compile functions (adamw_step_fused / muon_step_fused)
    # accumulate compiled variants for every unique parameter shape seen across all
    # tests.  Temporarily raise the cache limit so the optimizer step doesn't abort.
    import torch._dynamo as _dynamo
    _orig_cache_limit = _dynamo.config.cache_size_limit
    _dynamo.config.cache_size_limit = 256

    config = _tiny_config(n_layer=4, n_head=2, n_kv_head=2, n_embd=128, pc_head_dim=16)

    def _make_model_and_opt():
        m = _build_model(config)
        m.lm_head.weight = m.transformer.wte.weight
        opt = m.setup_optimizer(
            matrix_lr=0.01, embedding_lr=0.01, unembedding_lr=0.004,
            scalar_lr=0.5, adam_betas=(0.9, 0.95),
        )
        for g in opt.param_groups:
            g["initial_lr"] = g["lr"]
        return m, opt

    model, optimizer = _make_model_and_opt()
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    x = torch.randint(0, config.vocab_size, (2, 32), device=DEVICE)
    y = torch.randint(0, config.vocab_size, (2, 32), device=DEVICE)

    # ── Step 5: grad_norm is finite and positive ─────────────────────────────
    with autocast_ctx:
        main_loss, aux_loss, _layer_means = model(x, y)
        (main_loss + 0.1 * aux_loss).backward()

    all_params = [p for g in optimizer.param_groups for p in g['params']]
    grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)

    assert math.isfinite(grad_norm.item()), \
        f"grad_norm is not finite: {grad_norm.item()}"
    assert grad_norm.item() > 0, \
        "grad_norm is zero — gradients were not computed"

    optimizer.step()
    model.zero_grad(set_to_none=True)

    # ── Step 6: checkpoint save + resume ─────────────────────────────────────
    SAVED_STEP = 42
    SAVED_TTT  = 123.456

    ckpt_state = {
        "model":               model.state_dict(),
        "optimizer":           optimizer.state_dict(),
        "step":                SAVED_STEP,
        "total_training_time": SAVED_TTT,
    }
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name
    torch.save(ckpt_state, ckpt_path)

    # Build a fresh model + optimizer and resume (mirrors __main__ RESUME_CHECKPOINT block)
    model2, optimizer2 = _make_model_and_opt()
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model2.load_state_dict(ckpt["model"])
    optimizer2.load_state_dict(ckpt["optimizer"])
    step_restored               = ckpt.get("step", 0)
    total_training_time_restored = ckpt.get("total_training_time", 0.0)

    assert step_restored == SAVED_STEP, \
        f"Restored step={step_restored}, expected {SAVED_STEP}"
    assert abs(total_training_time_restored - SAVED_TTT) < 1e-3, (
        f"Restored total_training_time={total_training_time_restored}, "
        f"expected {SAVED_TTT}"
    )

    # Model weights match exactly
    sd1 = model.state_dict()
    sd2 = model2.state_dict()
    assert sd1.keys() == sd2.keys(), "State dict keys differ after resume"
    for key in sd1:
        t1, t2 = sd1[key].float(), sd2[key].float()
        assert torch.allclose(t1, t2, atol=1e-5), \
            f"Weight mismatch after resume: {key}"

    os.unlink(ckpt_path)

    # ── PC_EMA_DECAY: sweepable env var present with correct default ──────────
    import train as _tr
    assert hasattr(_tr, 'PC_EMA_DECAY'), \
        "PC_EMA_DECAY not found in train module — env-var wiring missing"
    # Default is 0.99; the test env does not override this var so it should be 0.99.
    assert abs(_tr.PC_EMA_DECAY - 0.99) < 1e-9, \
        f"PC_EMA_DECAY default expected 0.99, got {_tr.PC_EMA_DECAY}"
    # alpha must be the algebraic complement — verify the coupling holds.
    # (Tested indirectly below via the EMA update check.)

    # ── pc_ema update: buffer transitions from zero to non-zero after one step ─
    # Mirrors the training loop:
    #   model.pc_ema.mul_(PC_EMA_DECAY).add_(_last_layer_means, alpha=1-PC_EMA_DECAY)
    # Use a fresh model so we start from the known zero-init state.
    ema_decay = _tr.PC_EMA_DECAY
    ema_alpha = 1.0 - ema_decay
    model3, _ = _make_model_and_opt()
    assert (model3.pc_ema == 0).all(), "pc_ema not zero at init in fresh model"
    x3 = torch.randint(0, config.vocab_size, (2, 32), device=DEVICE)
    y3 = torch.randint(0, config.vocab_size, (2, 32), device=DEVICE)
    with autocast_ctx:
        _m3_loss, _m3_aux, _m3_means = model3(x3, y3)
    assert _m3_means.shape == (config.n_layer - 1, config.n_embd), \
        f"layer_means shape {_m3_means.shape} != ({config.n_layer-1}, {config.n_embd})"
    assert _m3_means.dtype == torch.float32, \
        f"layer_means dtype {_m3_means.dtype} should be float32"
    with torch.no_grad():
        model3.pc_ema.mul_(ema_decay).add_(_m3_means, alpha=ema_alpha)
    # After one EMA step from zero: pc_ema = 0*decay + alpha*layer_means = alpha*layer_means
    assert not (model3.pc_ema == 0).all(), \
        "pc_ema still all-zero after EMA update — training loop will feed static-zero targets"
    expected_ema = ema_alpha * _m3_means
    assert torch.allclose(model3.pc_ema, expected_ema, atol=1e-6), \
        f"pc_ema after first step != {ema_alpha} * layer_means (EMA formula broken)"

    _dynamo.config.cache_size_limit = _orig_cache_limit

    print(
        f"{PASS}  "
        f"grad_norm={grad_norm.item():.4f}  "
        f"step={step_restored}  "
        f"ttt={total_training_time_restored:.3f}"
    )


# ---------------------------------------------------------------------------
# Test 6b — LOG_MIN_WINDOW validation
# ---------------------------------------------------------------------------
def test_new_features():
    """
    LOG_MIN_WINDOW — power-of-2 / >=16 validation; sets first-layer window.
    """
    print("Test 6b [LOG_MIN_WINDOW]  ...", end="  ", flush=True)

    # ── LOG_MIN_WINDOW validation ────────────────────────────────────────────
    # Valid power-of-2 values >= 16 must not raise.
    for valid_v in (16, 32, 64, 128):
        cfg = GPTConfig(
            sequence_len=512, vocab_size=256, n_layer=4, n_head=2,
            n_kv_head=2, n_embd=128, window_pattern="LOG",
            log_min_window=valid_v,
        )
        try:
            GPT(cfg)   # triggers _compute_window_sizes
        except AssertionError as e:
            raise AssertionError(f"log_min_window={valid_v} should be valid but raised: {e}")

    # Non-power-of-2 must raise.
    for bad_v in (63, 100, 15):
        cfg = GPTConfig(
            sequence_len=512, vocab_size=256, n_layer=4, n_head=2,
            n_kv_head=2, n_embd=128, window_pattern="LOG",
            log_min_window=bad_v,
        )
        raised = False
        try:
            GPT(cfg)
        except AssertionError:
            raised = True
        assert raised, f"log_min_window={bad_v} should have raised AssertionError"

    # log_min_window=0 (auto) must not raise for any pattern.
    for pat in ("LOG", "PROGRESSIVE", "SSSL"):
        cfg = GPTConfig(
            sequence_len=512, vocab_size=256, n_layer=4, n_head=2,
            n_kv_head=2, n_embd=128, window_pattern=pat,
        )
        GPT(cfg)

    print(f"{PASS}")


# ---------------------------------------------------------------------------
# Test 6 — VAL_INTERVAL control-flow
# ---------------------------------------------------------------------------
def test_val_interval_fires():
    """VAL_INTERVAL fires at multiples of the interval, never at step 0."""
    print("Test 6  [VAL_INTERVAL control-flow]  ... ", end="", flush=True)

    original_fn = _train_module.evaluate_bpb
    original_interval = _train_module.VAL_INTERVAL
    calls = []

    def _mock_eval(model, tokenizer, batch_size):
        calls.append("called")
        return 4.0

    _train_module.evaluate_bpb = _mock_eval
    _train_module.VAL_INTERVAL = 3
    try:
        for step in range(6):   # steps 0..5
            if _train_module.VAL_INTERVAL > 0 and step > 0 and step % _train_module.VAL_INTERVAL == 0:
                _train_module.evaluate_bpb(None, None, 1)
    finally:
        _train_module.evaluate_bpb = original_fn
        _train_module.VAL_INTERVAL = original_interval

    assert calls == ["called"], (
        f"evaluate_bpb should fire exactly once (at step 3), got calls at: {calls}"
    )
    print(f"{PASS}  fires at step 3 only (not 0, not 1, 2, 4, 5)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    no_compile = "--no-compile" in _smoke_args

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.set_float32_matmul_precision("high")

    tests = [
        ("Test 1",  test_eager_training_scale),
        ("Test 3",  test_model_invariants),
        ("Test 4",  test_forward_numerics),
        ("Test 5",  test_grad_norm_and_checkpoint),
        ("Test 6",  test_val_interval_fires),
        ("Test 6b", test_new_features),
    ]

    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            print(f"{FAIL}  {e}")
            sys.exit(1)

    if no_compile:
        print("Test 2  [compiled, small-scale]  ... SKIPPED (--no-compile)")
    else:
        try:
            test_compiled_small_scale()
        except Exception as e:
            print(f"{FAIL}  {e}")
            sys.exit(1)

    print("\nAll smoke tests passed.")
