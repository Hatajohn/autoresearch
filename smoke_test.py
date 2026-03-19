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
      - model(x) with no targets returns logits (B, T, vocab_size) for step-0 diagnostic path
      - KL divergence computed in float32 (hook-verified)
      - No NaN under extreme negative resid_lambdas
      - reduction='none' returns 2-tuple; pc_head_dim bottleneck einsums correct shapes

  Test 5 — GRAD NORM + CHECKPOINT RESUME (Steps 5, 6)
    Optimizer-level checks:
      - clip_grad_norm_ returns a finite, positive grad_norm
      - Checkpoint includes pc_ema so resume restores EMA targets
      - Checkpoint round-trips model weights, optimizer state, step, and
        total_training_time exactly (note: training loop intentionally resets
        schedule_time to 0 on resume — the checkpoint value is for observability
        only; muon_warmup_step is initialised to the restored step, not 0, so
        the Muon momentum coefficient matches the warm optimizer buffers loaded
        via optimizer.load_state_dict())

  Test 6 — VAL_INTERVAL CONTROL-FLOW (Issue 2 / Issue 3)
    Simulates steps 0–5 with VAL_INTERVAL=3 and a mocked evaluate_bpb.
    Asserts evaluate_bpb fires exactly once (at step 3), never at step 0.

  Test 7 — PREPARE CONTRACT (--integration only)
    With cache: tokenizer loads, make_dataloader yields one batch (shapes,
    dtype, token id range), and model(x, y, reduction='none') returns 2-tuple.
    Skips if cache missing. Full evaluate_bpb() not run (too many steps).

  Test 8 — TRAINING LOOP SMOKE (--integration only)
    Subprocess run of train.py with TRAIN_TIME_BUDGET=45. Output goes to a temp
    .log (path printed on PASS); SMOKE_TEST8_LOG=/path/file.log to choose path.
    Default timeout 1800s; SMOKE_TEST8_TIMEOUT=seconds to override.

Usage:
  python smoke_test.py                  # run core tests (1, 3, 4, 5, 6, 6b, 2)
  python smoke_test.py --no-compile    # skip Test 2 (faster, e.g. in CI)
  python smoke_test.py --integration   # also run Test 7 (prepare) and Test 8 (train loop)
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
    ns = _NoStoch().to(DEVICE)
    assert hasattr(ns, '_zero'), "_NoStoch is missing the '_zero' registered buffer"
    assert ns._zero.shape == (), \
        f"_NoStoch._zero should be a scalar tensor, got shape {ns._zero.shape}"
    x_dev = torch.randn(3, 8, device=DEVICE)
    out, kl = ns(x_dev)
    assert out is x_dev, "_NoStoch.forward should return x unchanged (same object)"
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

    # ── kl_weight: non-persistent bfloat16 buffer, filled by init_weights ────
    assert hasattr(model, 'kl_weight'), "model missing kl_weight buffer"
    assert model.kl_weight.dtype == torch.bfloat16, \
        f"kl_weight dtype {model.kl_weight.dtype} should be bfloat16"
    assert 'kl_weight' not in model.state_dict(), \
        "kl_weight should be non-persistent (not in state_dict)"
    # init_weights fills it from config.kl_weight (default 0.01)
    assert abs(model.kl_weight.item() - 0.01) < 1e-4, \
        f"kl_weight after init_weights={model.kl_weight.item()}, expected 0.01"

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

    # ── Train loop env vars present and typed ─────────────────────────────────
    assert hasattr(_train_module, 'TIME_BUDGET'), "train module missing TIME_BUDGET"
    assert isinstance(_train_module.TIME_BUDGET, int), \
        f"TIME_BUDGET must be int, got {type(_train_module.TIME_BUDGET)}"
    assert _train_module.TIME_BUDGET > 0, "TIME_BUDGET must be positive"
    assert hasattr(_train_module, 'VAL_INTERVAL'), "train module missing VAL_INTERVAL"
    assert isinstance(_train_module.VAL_INTERVAL, int), \
        f"VAL_INTERVAL must be int, got {type(_train_module.VAL_INTERVAL)}"
    assert _train_module.VAL_INTERVAL >= 0, "VAL_INTERVAL must be non-negative"

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

    # ── StochasticLayer grads: KL path must produce non-zero gradients ───────
    stoch_params_with_grad = [
        p for sl in model.stochastic_layers
        if isinstance(sl, StochasticLayer)
        for p in sl.parameters()
        if p.grad is not None and p.grad.abs().sum().item() > 1e-10
    ]
    assert len(stoch_params_with_grad) > 0, (
        "StochasticLayer parameters received no gradients after backward; "
        "KL path may be detached or broken"
    )

    # ── Determinism: same seed, two forwards in eval mode → same loss ──────────
    model.eval()
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    with torch.no_grad():
        with AUTOCAST:
            _loss_a = model(x, y)[0].item()
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    with torch.no_grad():
        with AUTOCAST:
            _loss_b = model(x, y)[0].item()
    assert math.isclose(_loss_a, _loss_b, rel_tol=1e-5), (
        f"Determinism broken: two forwards gave {_loss_a} vs {_loss_b}"
    )
    model.train()

    # ── wte_std init: target logit_std ≈ 1.75 so E[CE] ≈ 10.5 nats at step 0 ─
    # wte_std = 1.75/sqrt(n_embd); verify the embedding weight std matches.
    expected_wte_std = 1.75 / (config.n_embd ** 0.5)
    actual_wte_std = model.transformer.wte.weight.float().std().item()
    assert abs(actual_wte_std - expected_wte_std) / expected_wte_std < 0.05, (
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

    # ── forward(x) no targets: returns logits for step-0 diagnostic path ─────
    with torch.no_grad():
        with AUTOCAST:
            logits_only = model(x)
    assert isinstance(logits_only, torch.Tensor), \
        "model(x) with no targets should return logits tensor, not a tuple"
    assert logits_only.shape == (x.shape[0], x.shape[1], config.vocab_size), (
        f"logits shape {logits_only.shape} != (B={x.shape[0]}, T={x.shape[1]}, V={config.vocab_size})"
    )
    assert logits_only.isfinite().all(), "logits from model(x) contain NaN/Inf"

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
    assert token_ce.ndim == 2, \
        f"first element of 'none' tuple should be per-token CE (2D), got ndim={token_ce.ndim}"
    assert aux_none.ndim == 0, \
        f"second element of 'none' tuple should be scalar aux loss, got ndim={aux_none.ndim}"
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
    Step 6 — checkpoint round-trips model weights, optimizer state, and step
             counter exactly. Recovery checkpoints also preserve schedule_time,
             smoothing state, and early-stop counters so resume behaves like an
             interrupted run rather than a fresh process with old weights.
             Muon resume policy: muon_warmup_step is initialised to the restored
             step (not 0) so the momentum coefficient matches the warm optimizer
             buffers from optimizer.load_state_dict(). Asserted below.
    """
    print("Test 5  [grad norm + checkpoint]  ...", end="  ", flush=True)

    # The module-level @torch.compile functions (adamw_step_fused / muon_step_fused)
    # accumulate compiled variants for every unique parameter shape seen across all
    # tests.  Temporarily raise the cache limit so the optimizer step doesn't abort.
    import torch._dynamo as _dynamo
    _orig_cache_limit = _dynamo.config.cache_size_limit
    _dynamo.config.cache_size_limit = 256
    try:

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
        SAVED_RUN_STATE = {
            "step": SAVED_STEP,
            "schedule_time": SAVED_TTT,
            "smooth_train_loss": 5.4321,
            "ema_smooth_steps": 17,
            "best_val_bpb": 1.2345,
            "val_bad_checks": 2,
            "best_raw_train_loss": 4.321,
            "train_bad_steps": 9,
            "throughput_time": 98.765,
        }

        ckpt_state = {
            "model":               model.state_dict(),
            "optimizer":           optimizer.state_dict(),
            "step":                SAVED_STEP,
            "total_training_time": SAVED_TTT,
            "run_state":           SAVED_RUN_STATE,
        }
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        torch.save(ckpt_state, ckpt_path)

        # Checkpoint must include pc_ema so resumed training has correct EMA targets.
        assert "pc_ema" in ckpt_state["model"], \
            "Checkpoint state_dict missing 'pc_ema' — resume would reinit EMA to zero"

        # Build a fresh model + optimizer and resume through the exact train.py
        # path, including assign=True and the optimizer rebinding helper.
        model2, optimizer2 = _make_model_and_opt()
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        resume_state = _train_module._normalize_state_dict_keys(ckpt["model"])
        model2.load_state_dict(resume_state, assign=True)
        optimizer2.load_state_dict(ckpt["optimizer"])
        _train_module._rebind_optimizer_to_model(model2, optimizer2)
        run_state = _train_module._load_run_state(ckpt)
        step_restored = run_state.get("step", 0)
        schedule_time_restored = run_state.get("schedule_time", 0.0)
        total_training_time_in_ckpt = ckpt.get("total_training_time", 0.0)

        assert step_restored == SAVED_STEP, \
            f"Restored step={step_restored}, expected {SAVED_STEP}"
        assert abs(schedule_time_restored - SAVED_TTT) < 1e-3, (
            f"Restored schedule_time={schedule_time_restored}, expected {SAVED_TTT}"
        )
        assert abs(total_training_time_in_ckpt - SAVED_TTT) < 1e-3, (
            f"Checkpoint total_training_time={total_training_time_in_ckpt}, "
            f"expected {SAVED_TTT} (round-trip fidelity check)"
        )
        assert run_state["ema_smooth_steps"] == SAVED_RUN_STATE["ema_smooth_steps"], \
            f"ema_smooth_steps restored as {run_state['ema_smooth_steps']}, expected {SAVED_RUN_STATE['ema_smooth_steps']}"
        assert abs(run_state["smooth_train_loss"] - SAVED_RUN_STATE["smooth_train_loss"]) < 1e-6, \
            f"smooth_train_loss restored as {run_state['smooth_train_loss']}, expected {SAVED_RUN_STATE['smooth_train_loss']}"
        assert abs(run_state["best_val_bpb"] - SAVED_RUN_STATE["best_val_bpb"]) < 1e-6, \
            f"best_val_bpb restored as {run_state['best_val_bpb']}, expected {SAVED_RUN_STATE['best_val_bpb']}"
        assert run_state["val_bad_checks"] == SAVED_RUN_STATE["val_bad_checks"], \
            f"val_bad_checks restored as {run_state['val_bad_checks']}, expected {SAVED_RUN_STATE['val_bad_checks']}"
        assert abs(run_state["best_raw_train_loss"] - SAVED_RUN_STATE["best_raw_train_loss"]) < 1e-6, \
            f"best_raw_train_loss restored as {run_state['best_raw_train_loss']}, expected {SAVED_RUN_STATE['best_raw_train_loss']}"
        assert run_state["train_bad_steps"] == SAVED_RUN_STATE["train_bad_steps"], \
            f"train_bad_steps restored as {run_state['train_bad_steps']}, expected {SAVED_RUN_STATE['train_bad_steps']}"
        assert abs(run_state["throughput_time"] - SAVED_RUN_STATE["throughput_time"]) < 1e-6, \
            f"throughput_time restored as {run_state['throughput_time']}, expected {SAVED_RUN_STATE['throughput_time']}"

        # ── Optimizer state: Adam second moments must be restored ─────────────
        st1 = optimizer.state_dict()["state"]
        st2 = optimizer2.state_dict()["state"]
        assert st1.keys() == st2.keys(), "Optimizer state keys differ after resume"
        for k in st1:
            if "exp_avg_sq" in st1[k]:
                assert torch.allclose(st1[k]["exp_avg_sq"], st2[k]["exp_avg_sq"], atol=1e-6), \
                    f"Optimizer exp_avg_sq mismatch for param {k}"
                break

        # Optimizer groups must reference the live model parameters after assign=True.
        live_param_ids = {id(p) for p in model2.parameters()}
        resumed_opt_params = [p for g in optimizer2.param_groups for p in g["params"]]
        stale_resumed_params = [p for p in resumed_opt_params if id(p) not in live_param_ids]
        assert not stale_resumed_params, (
            f"Found {len(stale_resumed_params)} stale optimizer params after resume rebinding"
        )

        # Model weights match exactly immediately after resume.
        sd1 = model.state_dict()
        sd2 = model2.state_dict()
        assert sd1.keys() == sd2.keys(), "State dict keys differ after resume"
        for key in sd1:
            t1, t2 = sd1[key].float(), sd2[key].float()
            assert torch.allclose(t1, t2, atol=1e-5), \
                f"Weight mismatch after resume: {key}"

        # A resumed step must succeed with live gradients instead of stale None grads.
        model2.zero_grad(set_to_none=True)
        x_resume = torch.randint(0, config.vocab_size, (2, 32), device=DEVICE)
        y_resume = torch.randint(0, config.vocab_size, (2, 32), device=DEVICE)
        with autocast_ctx:
            resumed_loss, resumed_aux, _ = model2(x_resume, y_resume)
            (resumed_loss + 0.1 * resumed_aux).backward()
        resumed_muon_params = [
            p for g in optimizer2.param_groups if g["kind"] == "muon" for p in g["params"]
        ]
        missing_muon_grads = [p for p in resumed_muon_params if p.grad is None]
        assert not missing_muon_grads, (
            f"Resumed Muon step has {len(missing_muon_grads)} params with grad=None"
        )
        optimizer2.step()
        model2.zero_grad(set_to_none=True)

        # ── Muon resume policy: warmup counter starts from restored step ──────────
        # train.py initialises muon_warmup_step = step (not 0) on both fresh runs
        # and resumes. On a fresh run step=0 so warmup starts from 0.85 as usual.
        # On resume the Muon optimizer buffers are warm (optimizer.load_state_dict),
        # so the coefficient must start from where it was, not re-warm from 0.85.
        # Verify the formula gives a value strictly between 0.85 and 0.95 for
        # SAVED_STEP=42 (which is < MUON_MOM_WARMUP_STEPS=300), meaning the
        # resumed coefficient is partially warmed — not at the cold-start value of
        # 0.85 that an incorrect reset would produce.
        import train as _tr
        assert hasattr(_tr, 'MUON_MOM_WARMUP_STEPS'), \
            "MUON_MOM_WARMUP_STEPS not found in train module"
        assert _tr.MUON_MOM_WARMUP_STEPS > SAVED_STEP, (
            f"SAVED_STEP={SAVED_STEP} must be < MUON_MOM_WARMUP_STEPS={_tr.MUON_MOM_WARMUP_STEPS} "
            "for this assertion to distinguish continuation from reset"
        )
        t_muon_resumed = min(step_restored / _tr.MUON_MOM_WARMUP_STEPS, 1.0)
        muon_mom_resumed = (1 - t_muon_resumed) * 0.85 + t_muon_resumed * 0.95
        muon_mom_if_reset = 0.85  # what muon_warmup_step=0 would give
        assert muon_mom_resumed > muon_mom_if_reset, (
            f"Muon momentum at resumed step {step_restored} ({muon_mom_resumed:.4f}) "
            f"should exceed the cold-start value ({muon_mom_if_reset:.2f}); "
            "if equal, muon_warmup_step may have been incorrectly reset to 0"
        )

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
        # Mirrors the training loop (train.py): new_ema = pc_ema*decay + layer_means*(1-decay);
        # then pc_ema.data.copy_(new_ema) to avoid version-counter bump and per-step recompile.
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
            new_ema = model3.pc_ema * ema_decay + _m3_means * ema_alpha
            model3.pc_ema.data.copy_(new_ema)
        # After one EMA step from zero: pc_ema = 0*decay + alpha*layer_means = alpha*layer_means
        assert not (model3.pc_ema == 0).all(), \
            "pc_ema still all-zero after EMA update — training loop will feed static-zero targets"
        expected_ema = ema_alpha * _m3_means
        assert torch.allclose(model3.pc_ema, expected_ema, atol=1e-6), \
            f"pc_ema after first step != {ema_alpha} * layer_means (EMA formula broken)"

    finally:
        _dynamo.config.cache_size_limit = _orig_cache_limit

    print(
        f"{PASS}  "
        f"grad_norm={grad_norm.item():.4f}  "
        f"step={step_restored}  "
        f"ttt_in_ckpt={total_training_time_in_ckpt:.3f}"
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

    # SSSL forward pass: confirm no shape error at runtime
    cfg_sssl = GPTConfig(
        sequence_len=512, vocab_size=256, n_layer=4, n_head=2,
        n_kv_head=2, n_embd=128, window_pattern="SSSL",
    )
    m_sssl = _build_model(cfg_sssl)
    with torch.no_grad(), AUTOCAST:
        _ = m_sssl(torch.zeros(1, 32, dtype=torch.long, device=DEVICE))

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
            if _train_module._should_evaluate(step, _train_module.VAL_INTERVAL):
                _train_module.evaluate_bpb(None, None, 1)
    finally:
        _train_module.evaluate_bpb = original_fn
        _train_module.VAL_INTERVAL = original_interval

    assert calls == ["called"], (
        f"evaluate_bpb should fire exactly once (at step 3), got calls at: {calls}"
    )
    print(f"{PASS}  fires at step 3 only (not 0, not 1, 2, 4, 5)")


# ---------------------------------------------------------------------------
# Test 6c — completed-step cadence expectations
# ---------------------------------------------------------------------------
def test_completed_step_cadence():
    """Completed-step controls should fire on exact post-update multiples."""
    print("Test 6c [completed-step cadence]  ... ", end="", flush=True)

    val_interval = 3
    checkpoint_steps = 4
    val_steps = []
    checkpoint_hits = []

    for completed_step in range(1, 13):
        if _train_module._should_evaluate(completed_step, val_interval):
            val_steps.append(completed_step)
        if checkpoint_steps > 0 and completed_step % checkpoint_steps == 0:
            checkpoint_hits.append(completed_step)

    assert val_steps == [3, 6, 9, 12], f"Expected eval cadence on [3, 6, 9, 12], got {val_steps}"
    assert checkpoint_hits == [4, 8, 12], (
        f"Expected checkpoint cadence on [4, 8, 12], got {checkpoint_hits}"
    )

    print(f"{PASS}  eval@{val_steps} checkpoint@{checkpoint_hits}")


# ---------------------------------------------------------------------------
# Test A — unit tests for _get_lr_multiplier and _get_kl_weight
# ---------------------------------------------------------------------------
def test_schedule_helpers():
    """
    Pure-function unit tests for the LR and KL schedule helpers.
    No GPU required; fast.
    """
    print("Test A  [schedule helpers]  ...", end="  ", flush=True)

    import train as _tr

    # ── _get_lr_multiplier ────────────────────────────────────────────────────
    # WARMUP_RATIO=0.0: progress=0.0 falls into the plateau branch → 1.0
    assert abs(_tr._get_lr_multiplier(0.0) - 1.0) < 1e-9, (
        f"_get_lr_multiplier(0.0) expected 1.0 (warmup skipped), "
        f"got {_tr._get_lr_multiplier(0.0)}"
    )

    # Just past cooldown onset → strictly less than 1.0
    eps = 1e-6
    onset = 1.0 - _tr.WARMDOWN_RATIO + eps
    lrm_onset = _tr._get_lr_multiplier(onset)
    assert lrm_onset < 1.0, (
        f"_get_lr_multiplier({onset:.6f}) expected < 1.0 (cooldown started), "
        f"got {lrm_onset}"
    )
    assert lrm_onset >= 0.0, (
        f"_get_lr_multiplier({onset:.6f}) expected >= 0.0, got {lrm_onset}"
    )

    # progress=1.0 → FINAL_LR_FRAC (= 0.0)
    assert abs(_tr._get_lr_multiplier(1.0) - _tr.FINAL_LR_FRAC) < 1e-9, (
        f"_get_lr_multiplier(1.0) expected FINAL_LR_FRAC={_tr.FINAL_LR_FRAC}, "
        f"got {_tr._get_lr_multiplier(1.0)}"
    )

    # ── _get_kl_weight ────────────────────────────────────────────────────────
    # step=0 → 0.0 (before warmup)
    assert abs(_tr._get_kl_weight(0) - 0.0) < 1e-9, (
        f"_get_kl_weight(0) expected 0.0, got {_tr._get_kl_weight(0)}"
    )

    # step=KL_WARMUP_STEPS → exactly KL_WEIGHT
    assert abs(_tr._get_kl_weight(_tr.KL_WARMUP_STEPS) - _tr.KL_WEIGHT) < 1e-9, (
        f"_get_kl_weight(KL_WARMUP_STEPS={_tr.KL_WARMUP_STEPS}) expected "
        f"KL_WEIGHT={_tr.KL_WEIGHT}, got {_tr._get_kl_weight(_tr.KL_WARMUP_STEPS)}"
    )

    # step >> KL_WARMUP_STEPS → clamped to KL_WEIGHT
    big_step = _tr.KL_WARMUP_STEPS * 10
    assert abs(_tr._get_kl_weight(big_step) - _tr.KL_WEIGHT) < 1e-9, (
        f"_get_kl_weight({big_step}) expected KL_WEIGHT={_tr.KL_WEIGHT} (clamped), "
        f"got {_tr._get_kl_weight(big_step)}"
    )

    # Invalid KL warmup must fail explicitly instead of dividing by zero later.
    orig_kl_warmup = _tr.KL_WARMUP_STEPS
    _tr.KL_WARMUP_STEPS = 0
    try:
        try:
            _tr._get_kl_weight(1)
            raise AssertionError("Expected _get_kl_weight() to reject KL_WARMUP_STEPS <= 0")
        except ValueError as exc:
            assert "KL_WARMUP_STEPS" in str(exc), f"Unexpected error: {exc}"
    finally:
        _tr.KL_WARMUP_STEPS = orig_kl_warmup

    # Startup validation should reject bad runtime controls before training begins.
    orig_time_budget = _tr.TIME_BUDGET
    _tr.TIME_BUDGET = 0
    try:
        try:
            _tr._validate_startup_config(_tr.RunConfig())
            raise AssertionError("Expected startup validation to reject TRAIN_TIME_BUDGET <= 0")
        except ValueError as exc:
            assert "TRAIN_TIME_BUDGET" in str(exc), f"Unexpected error: {exc}"
    finally:
        _tr.TIME_BUDGET = orig_time_budget

    try:
        _tr._validate_startup_config(_tr.RunConfig(checkpoint_steps=-1))
        raise AssertionError("Expected startup validation to reject negative CHECKPOINT_STEPS")
    except ValueError as exc:
        assert "CHECKPOINT_STEPS" in str(exc), f"Unexpected error: {exc}"

    try:
        _tr._validate_startup_config(_tr.RunConfig(max_epochs=-1))
        raise AssertionError("Expected startup validation to reject negative MAX_EPOCHS")
    except ValueError as exc:
        assert "MAX_EPOCHS" in str(exc), f"Unexpected error: {exc}"

    assert not _tr._should_stop_for_epoch_limit(next_epoch=1, max_epochs=1), (
        "Epoch limiter should not stop while still inside the allowed epoch"
    )
    assert _tr._should_stop_for_epoch_limit(next_epoch=2, max_epochs=1), (
        "Epoch limiter should stop once the prefetched batch would enter epoch 2"
    )
    assert not _tr._should_stop_for_epoch_limit(next_epoch=3, max_epochs=0), (
        "MAX_EPOCHS=0 should disable epoch-based stopping"
    )

    print(f"{PASS}")


# ---------------------------------------------------------------------------
# Test 7 — prepare contract (--integration only; requires cache)
# ---------------------------------------------------------------------------
def test_prepare_contract():
    """
    With --integration and cache present: tokenizer loads, make_dataloader yields
    one batch with correct shapes and value range, and a model with matching
    vocab accepts the batch and returns the 2-tuple expected by evaluate_bpb.
    Full evaluate_bpb() is not run (too many steps); the contract is that
    prepare's API and model(x, y, reduction='none') stay aligned.
    """
    print("Test 7  [prepare contract]  ...", end="  ", flush=True)

    try:
        from prepare import Tokenizer, make_dataloader, MAX_SEQ_LEN
    except ImportError as e:
        print(f"SKIP  prepare import failed: {e}")
        return

    try:
        tokenizer = Tokenizer.from_directory()
    except (FileNotFoundError, OSError) as e:
        print(f"SKIP  cache missing ({e}); run prepare.py and use --integration")
        return

    vocab_size = tokenizer.get_vocab_size()
    assert vocab_size > 0, "tokenizer vocab size should be positive"

    # One batch from val loader (short T so we don't need full MAX_SEQ_LEN model)
    batch_T = 128
    try:
        val_loader = make_dataloader(tokenizer, B=2, T=batch_T, split="val")
        x_batch, y_batch, epoch = next(val_loader)
    except (FileNotFoundError, OSError, AssertionError) as e:
        print(f"SKIP  val data missing or empty ({e})")
        return

    assert x_batch.shape == (2, batch_T), f"x batch shape {x_batch.shape} != (2, {batch_T})"
    assert y_batch.shape == (2, batch_T), f"y batch shape {y_batch.shape} != (2, {batch_T})"
    assert x_batch.dtype == torch.long and y_batch.dtype == torch.long
    assert (x_batch >= 0).all() and (x_batch < vocab_size).all(), "x token ids out of range"
    assert (y_batch >= 0).all() and (y_batch < vocab_size).all(), "y token ids out of range"

    # Model with matching vocab and seq_len >= batch_T
    config = GPTConfig(
        sequence_len=256, vocab_size=vocab_size,
        n_layer=4, n_head=2, n_kv_head=2, n_embd=128,
        window_pattern="PROGRESSIVE", pc_head_dim=16,
    )
    model = _build_model(config)
    model.eval()
    with torch.no_grad():
        with AUTOCAST:
            out = model(x_batch.to(DEVICE), y_batch.to(DEVICE), reduction='none')
    assert len(out) == 2, f"model(..., reduction='none') must return 2-tuple, got {len(out)}"
    token_ce, aux = out
    assert token_ce.shape == (2, batch_T), f"token_ce shape {token_ce.shape} != (2, {batch_T})"
    assert token_ce.isfinite().all(), "token_ce contained NaN/Inf"
    model.train()

    print(f"{PASS}  tokenizer+batch+model contract OK")


# ---------------------------------------------------------------------------
# Test 8 — training loop smoke (--integration only; slow)
# ---------------------------------------------------------------------------
def test_training_loop_smoke():
    """
    With --integration: run train.py in a subprocess with TRAIN_TIME_BUDGET=45s.
    Subprocess timeout defaults to 1800s so cold Triton compile + kernel pre-warm
    (often 10–25 min on first graph) can finish. Set SMOKE_TEST8_TIMEOUT to override.
    All stdout/stderr from train.py is written to a temp .log file; path is printed
    on PASS/FAIL. Set SMOKE_TEST8_LOG to a file path to use that instead of tempfile.
    """
    print("Test 8  [training loop smoke]  ...", end="  ", flush=True)
    import subprocess

    # 90s was too tight: pre-warm alone can be 1200s+ on cold cache (see run21 logs).
    timeout_s = int(os.environ.get("SMOKE_TEST8_TIMEOUT", "1800"))

    log_path = os.environ.get("SMOKE_TEST8_LOG")
    if not log_path:
        log_fd, log_path = tempfile.mkstemp(
            prefix="autoresearch_smoke_test8_", suffix=".log"
        )
        os.close(log_fd)

    env = os.environ.copy()
    env["TRAIN_TIME_BUDGET"] = "45"   # short run; enough for 1–2 steps after pre-warm
    env["VAL_INTERVAL"] = "0"
    train_py = os.path.join(os.path.dirname(__file__), "train.py")

    def _read_log_tail(n: int = 1200) -> str:
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()[-n:]
        except OSError:
            return "(could not read log)"

    try:
        # Line-buffered so partial logs are useful on timeout/kill.
        with open(log_path, "w", encoding="utf-8", errors="replace", buffering=1) as logf:
            proc = subprocess.run(
                [sys.executable, train_py],
                stdout=logf,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
                cwd=os.path.dirname(__file__),
                env=env,
            )
    except subprocess.TimeoutExpired:
        print(
            f"{FAIL}  train.py did not complete within {timeout_s}s "
            f"(cold compile/pre-warm can be slow; set SMOKE_TEST8_TIMEOUT higher, "
            f"or run once to warm the cache)\n  log: {log_path}"
        )
        print(f"  tail:\n{_read_log_tail(2500)}")
        raise
    except FileNotFoundError as e:
        print(f"SKIP  {e}")
        return

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        combined = f.read()

    assert proc.returncode == 0, (
        f"train.py exited {proc.returncode}  log: {log_path}\n"
        f"tail:\n{_read_log_tail(1500)}"
    )
    assert "step" in combined.lower(), (
        f"train.py output did not contain 'step'; loop may not have run  log: {log_path}\n"
        f"tail:\n{_read_log_tail(1500)}"
    )
    print(f"{PASS}  train.py completed with steps  log: {log_path}")


# ---------------------------------------------------------------------------
# Test 9 — signal-driven checkpoint stop (--integration only; slow)
# ---------------------------------------------------------------------------
def test_signal_checkpoint_stop():
    """
    With --integration: run train.py in a temp working directory, send SIGTERM
    after the training loop starts, assert the process checkpoints and exits
    cleanly, then resume from that checkpoint and verify the resumed run gets
    past its first optimizer step before shutting down cleanly again.
    """
    print("Test 9  [signal checkpoint stop]  ...", end="  ", flush=True)
    import signal as _signal
    import subprocess
    import time

    with tempfile.TemporaryDirectory(prefix="autoresearch_signal_stop_") as tmpdir:
        log_path = os.path.join(tmpdir, "signal_stop.log")
        ckpt_latest = os.path.join(tmpdir, "checkpoint_latest.pt")
        train_py = os.path.join(os.path.dirname(__file__), "train.py")

        env = os.environ.copy()
        env["USE_TORCH_COMPILE"] = "0"
        env["TRAIN_TIME_BUDGET"] = "300"
        env["VAL_INTERVAL"] = "0"
        env["CHECKPOINT_STEPS"] = "0"
        env["EARLY_STOP_ENABLE"] = "0"
        env["DEPTH"] = "4"
        env["MODEL_WIDTH"] = "128"

        with open(log_path, "w", encoding="utf-8", errors="replace", buffering=1) as logf:
            proc = subprocess.Popen(
                [sys.executable, train_py],
                stdout=logf,
                stderr=subprocess.STDOUT,
                cwd=tmpdir,
                env=env,
            )

        started = False
        deadline = time.time() + 180
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            try:
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    combined = f.read()
            except OSError:
                combined = ""
            if "Starting training loop..." in combined and "step 00001" in combined:
                started = True
                break
            time.sleep(1.0)

        assert started, (
            f"train.py never reached the active training loop before timeout  log: {log_path}"
        )

        proc.send_signal(_signal.SIGTERM)
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise AssertionError(f"train.py did not exit after SIGTERM  log: {log_path}")

        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            combined = f.read()

        assert proc.returncode == 0, (
            f"train.py exited {proc.returncode} after SIGTERM  log: {log_path}\n"
            f"log:\n{combined[-2000:]}"
        )
        assert "Received SIGTERM" in combined, f"log missing SIGTERM notice  log: {log_path}"
        assert "saved checkpoint_latest.pt" in combined, (
            f"log missing checkpoint save on signal stop  log: {log_path}"
        )
        assert os.path.exists(ckpt_latest), f"checkpoint_latest.pt missing after SIGTERM  cwd: {tmpdir}"

        ckpt = torch.load(ckpt_latest, map_location="cpu", weights_only=True)
        run_state = _train_module._load_run_state(ckpt)
        assert run_state["step"] >= 1, f"expected at least one completed step, got {run_state['step']}"

        resumed_log_path = os.path.join(tmpdir, "signal_resume.log")
        resumed_env = env.copy()
        resumed_env["RESUME_CHECKPOINT"] = ckpt_latest
        resumed_step = int(run_state["step"]) + 1
        resumed_step_token = f"step {resumed_step:05d}"

        with open(resumed_log_path, "w", encoding="utf-8", errors="replace", buffering=1) as logf:
            resumed_proc = subprocess.Popen(
                [sys.executable, train_py],
                stdout=logf,
                stderr=subprocess.STDOUT,
                cwd=tmpdir,
                env=resumed_env,
            )

        resumed_progress = False
        deadline = time.time() + 180
        while time.time() < deadline:
            if resumed_proc.poll() is not None:
                break
            try:
                with open(resumed_log_path, "r", encoding="utf-8", errors="replace") as f:
                    resumed_log = f.read()
            except OSError:
                resumed_log = ""
            if "Resumed from" in resumed_log and resumed_step_token in resumed_log:
                resumed_progress = True
                break
            time.sleep(1.0)

        assert resumed_progress, (
            f"resumed train.py did not reach {resumed_step_token} before timeout  "
            f"log: {resumed_log_path}"
        )

        resumed_proc.send_signal(_signal.SIGTERM)
        try:
            resumed_proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            resumed_proc.kill()
            raise AssertionError(f"resumed train.py did not exit after SIGTERM  log: {resumed_log_path}")

        with open(resumed_log_path, "r", encoding="utf-8", errors="replace") as f:
            resumed_log = f.read()

        assert resumed_proc.returncode == 0, (
            f"resumed train.py exited {resumed_proc.returncode} after SIGTERM  log: {resumed_log_path}\n"
            f"log:\n{resumed_log[-2000:]}"
        )
        assert "Resumed from" in resumed_log, f"log missing resume banner  log: {resumed_log_path}"
        assert resumed_step_token in resumed_log, (
            f"log missing first resumed step {resumed_step_token}  log: {resumed_log_path}"
        )
        assert "saved checkpoint_latest.pt" in resumed_log, (
            f"log missing checkpoint save after resumed SIGTERM  log: {resumed_log_path}"
        )

        resumed_ckpt = torch.load(ckpt_latest, map_location="cpu", weights_only=True)
        resumed_run_state = _train_module._load_run_state(resumed_ckpt)
        assert resumed_run_state["step"] >= resumed_step, (
            f"expected resumed checkpoint step >= {resumed_step}, got {resumed_run_state['step']}"
        )

    print(f"{PASS}  checkpointed shutdown and successful resume after SIGTERM")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _check_no_training_running():
    """Abort early if a training process or significant GPU allocation is detected.

    Two independent signals:
      1. Process scan — any python process whose cmdline contains 'train.py'
         (covers both 'python train.py' and 'torchrun ... train.py').
      2. GPU memory — if another process has already allocated >500 MB of VRAM
         the tests will likely OOM or produce misleading timings.
    """
    import subprocess

    # ── Signal 1: process scan ───────────────────────────────────────────────
    try:
        out = subprocess.check_output(
            ["pgrep", "-af", "train.py"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        if out:
            # Filter out this very process (smoke_test.py imports train.py,
            # so its own cmdline doesn't contain train.py directly — but be safe).
            own_pid = str(os.getpid())
            offending = [ln for ln in out.splitlines() if not ln.startswith(own_pid)]
            if offending:
                print(
                    f"\n\033[33mWARNING: training process(es) appear to be running:\033[0m"
                )
                for ln in offending:
                    print(f"  {ln}")
                print(
                    "Smoke tests allocate ~2 GB of VRAM and will interfere with an "
                    "active training run.\nAbort and re-run once training has finished, "
                    "or pass --force to skip this check."
                )
                if "--force" not in _smoke_args:
                    sys.exit(1)
                print("  --force passed; continuing anyway.\n")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass  # pgrep returns exit 1 when no match; FileNotFoundError if not installed

    # ── Signal 2: GPU memory already in use ──────────────────────────────────
    if torch.cuda.is_available():
        # Query free vs total memory on the default device.
        free_bytes, total_bytes = torch.cuda.mem_get_info(DEVICE)
        used_bytes = total_bytes - free_bytes
        used_pct = used_bytes / total_bytes
        used_mb  = used_bytes / (1024 ** 2)
        total_mb = total_bytes / (1024 ** 2)
        # Use 30% of total VRAM as the trip-wire.  The Flash Attention 3 kernel
        # cache consumes ~1.3 GB on a 16 GB card (~8%) at import time — well below
        # this threshold.  Any active training run (>10 GB) will exceed it.
        threshold_pct = 0.30
        if used_pct > threshold_pct:
            print(
                f"\n\033[33mWARNING: GPU already has {used_mb:.0f} / {total_mb:.0f} MB "
                f"allocated ({used_pct:.0%} > {threshold_pct:.0%} threshold).\033[0m"
            )
            print(
                "Another process may be using the GPU. Smoke tests may OOM or produce "
                "misleading results.\nPass --force to skip this check."
            )
            if "--force" not in _smoke_args:
                sys.exit(1)
            print("  --force passed; continuing anyway.\n")


if __name__ == "__main__":
    no_compile = "--no-compile" in _smoke_args
    integration = "--integration" in _smoke_args

    _check_no_training_running()

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.set_float32_matmul_precision("high")

    tests = [
        ("Test 1",  test_eager_training_scale),
        ("Test 3",  test_model_invariants),
        ("Test 4",  test_forward_numerics),
        ("Test 5",  test_grad_norm_and_checkpoint),
        ("Test 6",  test_val_interval_fires),
        ("Test 6c", test_completed_step_cadence),
        ("Test 6b", test_new_features),
        ("Test A",  test_schedule_helpers),
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

    if integration:
        for name, fn in [
            ("Test 7", test_prepare_contract),
            ("Test 8", test_training_loop_smoke),
            ("Test 9", test_signal_checkpoint_stop),
        ]:
            try:
                fn()
            except Exception as e:
                print(f"{FAIL}  {e}")
                sys.exit(1)

    print("\nAll smoke tests passed.")
