"""
Autoresearch pretraining script. Single-GPU.
Usage: uv run train.py

Structure:
  - Env and imports (Flash Attention, prepare, model, optimizer)
  - Run constants (time budget, resume, val interval) from env
  - Hyperparameters (architecture, optimization, PC)
  - main(): signals → torch setup → tokenizer → config → model → optimizer →
    compile → prewarm → training loop → final eval & checkpoint
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.expanduser("~/.cache/torchinductor"))

import gc
import math
import signal
import sys
import time
from dataclasses import asdict

import torch
import torch.nn.functional as F

# Flash Attention 3 (optional)
try:
    from kernels import get_kernel
    cap = torch.cuda.get_device_capability()
    repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
    fa3 = get_kernel(repo).flash_attn_interface
    print(f"Flash Attention 3 loaded (repo={repo}, cap={cap})", flush=True)
except Exception as e:
    print(f"Flash Attention 3 unavailable ({e}), falling back to torch SDPA.", flush=True)
    fa3 = None

import model as _model
_model.fa3 = fa3

from prepare import MAX_SEQ_LEN, Tokenizer, make_dataloader, evaluate_bpb

# Re-export for smoke_test and sample.py
from model import GPT, GPTConfig, _NoStoch, StochasticLayer, AUX_HORIZONS  # noqa: F401

# ---------------------------------------------------------------------------
# Run constants (env; no recompile when changed)
# ---------------------------------------------------------------------------

TIME_BUDGET = int(os.environ.get("TRAIN_TIME_BUDGET", "360"))
RESUME_CHECKPOINT = os.environ.get("RESUME_CHECKPOINT", "")
VAL_INTERVAL = int(os.environ.get("VAL_INTERVAL", "0"))

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

# Architecture
ASPECT_RATIO = 64
HEAD_DIM = 128
WINDOW_PATTERN = "LOG"
DEPTH = 8
DEVICE_BATCH_SIZE = 32
PC_HEAD_DIM = int(os.environ.get("PC_HEAD_DIM", str(max(64, 8 * ASPECT_RATIO // 8))))
LOG_MIN_WINDOW = int(os.environ.get("LOG_MIN_WINDOW", "0"))

# Optimization
TOTAL_BATCH_SIZE = 2**19
EMBEDDING_LR = 0.1
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.04
SCALAR_LR = 0.5
WEIGHT_DECAY = 0.2
ADAM_BETAS = (0.8, 0.95)
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = 0.5
FINAL_LR_FRAC = 0.0
LOG_SMOOTH_BETA = 0.9
LOGIT_SOFTCAP = 15

# Predictive coding (env-sweepable)
PC_WEIGHT = float(os.environ.get("PC_WEIGHT", "0.02"))
PC_FOCAL_GAMMA = float(os.environ.get("PC_FOCAL_GAMMA", "1.0"))
KL_WEIGHT = float(os.environ.get("KL_WEIGHT", "0.01"))
PC_DIAG_INTERVAL = int(os.environ.get("PC_DIAG_INTERVAL", "50"))
PC_EMA_DECAY = float(os.environ.get("PC_EMA_DECAY", "0.99"))

REFERENCE_BF16_PEAK_FLOPS = 989.5e12
KL_WARMUP_STEPS = 25
USE_STOCHASTIC_LAYERS = os.environ.get("USE_STOCHASTIC_LAYERS", "1") == "1"
# USE_TORCH_COMPILE: "1"/"full" = whole-model compile (best step speed, long cold start);
# "regional" = compile each transformer Block only (see PyTorch regional compilation recipe);
# "0"/"off" = no compile (fastest startup, slowest steps).
def _torch_compile_mode():
    v = os.environ.get("USE_TORCH_COMPILE", "1").strip().lower()
    if v in ("0", "false", "off", "no"):
        return "off"
    if v in ("regional", "blocks", "layer"):
        return "regional"
    return "full"


TORCH_COMPILE_MODE = _torch_compile_mode()


def _setup_signals():
    def _handle(signum, frame):
        name = getattr(signal.Signals(signum), "name", signum)
        print(f"\n[train] Received {name} — exiting cleanly.", flush=True)
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        sys.exit(0)
    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)


def _build_config(vocab_size):
    base_dim = DEPTH * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN,
        vocab_size=vocab_size,
        n_layer=DEPTH,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
        pc_head_dim=PC_HEAD_DIM,
        log_min_window=LOG_MIN_WINDOW,
        pc_focal_gamma=PC_FOCAL_GAMMA,
        kl_weight=KL_WEIGHT,
        logit_softcap=float(LOGIT_SOFTCAP),
        aux_horizons=AUX_HORIZONS,
        use_stochastic_layers=USE_STOCHASTIC_LAYERS,
    )


def _rebind_optimizer_to_compiled(model, optimizer):
    """After torch.compile, point optimizer at the compiled model's inner parameters."""
    inner = getattr(model, "_orig_mod", None)
    if inner is None:
        return
    param_lists = inner._get_optimizer_param_lists()
    assert len(param_lists) == len(optimizer.param_groups)
    old_flat = [p for g in optimizer.param_groups for p in g["params"]]
    new_flat = [p for lst in param_lists for p in lst]
    assert len(old_flat) == len(new_flat)
    for old_p, new_p in zip(old_flat, new_flat):
        if old_p in optimizer.state:
            optimizer.state[new_p] = optimizer.state.pop(old_p)
    for i, group in enumerate(optimizer.param_groups):
        group["params"] = param_lists[i]


def _get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    if progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    cooldown = (1.0 - progress) / WARMDOWN_RATIO
    return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC


def _get_kl_weight(step):
    return KL_WEIGHT * min(step / KL_WARMUP_STEPS, 1.0)


def main():
    _setup_signals()
    t_start = time.time()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.cache_size_limit = 64
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    # Tokenizer
    print("Loading tokenizer...", flush=True)
    t0 = time.time()
    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()
    t_tokenizer = time.time() - t0
    print(f"Vocab size: {vocab_size:,} ({t_tokenizer:.1f}s)", flush=True)

    config = _build_config(vocab_size)
    print(f"Model config: {asdict(config)}")

    # Model: meta → to_empty → init_weights
    print("Initializing model...", flush=True)
    t0 = time.time()
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    t_model_init = time.time() - t0
    print(f"  model init done ({t_model_init:.1f}s)", flush=True)

    model.lm_head.weight = model.transformer.wte.weight

    param_counts = model.num_scaling_params()
    print("Parameter counts:")
    for k, v in param_counts.items():
        print(f"  {k:24s}: {v:,}")
    num_flops_per_token = model.estimate_flops()
    print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

    tokens_per_step = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
    assert TOTAL_BATCH_SIZE % tokens_per_step == 0
    grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_step

    optimizer = model.setup_optimizer(
        unembedding_lr=UNEMBEDDING_LR,
        embedding_lr=EMBEDDING_LR,
        scalar_lr=SCALAR_LR,
        adam_betas=ADAM_BETAS,
        matrix_lr=MATRIX_LR,
        weight_decay=WEIGHT_DECAY,
    )

    step = 0
    total_training_time = 0.0
    if RESUME_CHECKPOINT:
        ckpt = torch.load(RESUME_CHECKPOINT, map_location=device)
        state = ckpt["model"]
        if any(k.startswith("_orig_mod.") for k in state.keys()):
            state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state, assign=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt.get("step", 0)
        # total_training_time stays 0.0 so this run's time budget starts when the loop begins
        print(f"Resumed from {RESUME_CHECKPOINT} at step {step}", flush=True)

    t0 = time.time()
    if TORCH_COMPILE_MODE == "full":
        print("Compiling full model (torch.compile)...", flush=True)
        model = torch.compile(model, dynamic=False, fullgraph=True)
        t_compile = time.time() - t0
        print(f"  torch.compile done ({t_compile:.1f}s)", flush=True)
        _rebind_optimizer_to_compiled(model, optimizer)
    elif TORCH_COMPILE_MODE == "regional":
        # https://docs.pytorch.org/tutorials/recipes/regional_compilation.html
        # Smaller graphs per Block → much shorter cold start than full-model compile;
        # step speed usually between eager and full compile.
        print("Regional compile: torch.compile each transformer Block...", flush=True)
        h = model.transformer.h
        for i in range(len(h)):
            h[i] = torch.compile(h[i], dynamic=False, fullgraph=True)
        t_compile = time.time() - t0
        print(f"  regional torch.compile setup ({t_compile:.1f}s; first fwd will finish kernel gen)", flush=True)
    else:
        t_compile = 0.0
        print("Skipping torch.compile (USE_TORCH_COMPILE=0) — fast startup, slower steps.", flush=True)

    # Pre-warm: dummy fwd+bwd + eval variant + step-0 diagnostic
    print("Pre-warming kernels...", flush=True)
    t0 = time.time()
    dummy_x = torch.zeros(DEVICE_BATCH_SIZE, MAX_SEQ_LEN, dtype=torch.long, device=device)
    dummy_y = torch.zeros(DEVICE_BATCH_SIZE, MAX_SEQ_LEN, dtype=torch.long, device=device)
    with autocast_ctx:
        d_loss, d_aux, _ = model(dummy_x, dummy_y)
        (d_loss + PC_WEIGHT * d_aux).backward()
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            model(dummy_x, dummy_y, reduction="none")
            diag_y = torch.randint(0, config.vocab_size, dummy_x.shape, device=device)
            step0_logits = model(dummy_x)
    print(f"  step-0 diagnostic: logit_std={step0_logits.float().std().item():.4f}, "
          f"raw_CE={F.cross_entropy(step0_logits.float().view(-1, step0_logits.size(-1)), diag_y.view(-1)).item():.4f} nats",
          flush=True)
    del dummy_x, dummy_y, diag_y, step0_logits, d_loss, d_aux
    torch.cuda.synchronize()
    t_prewarm = time.time() - t0
    print(f"  pre-warm done ({t_prewarm:.0f}s)", flush=True)

    torch.cuda.empty_cache()
    # After empty_cache(), this is model + optimizer only (no batch activations).
    print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated (model+optimizer after pre-warm)", flush=True)

    train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
    x, y, epoch = next(train_loader)

    print(f"Time budget: {TIME_BUDGET}s", flush=True)
    print(f"Grad accumulation steps: {grad_accum_steps}", flush=True)
    print("Starting training loop...", flush=True)

    all_params = [p for g in optimizer.param_groups for p in g["params"]]
    smooth_train_loss = 0.0
    # Debiased EMA must use steps since this process started smoothing, not global
    # step (resume resets smooth_train_loss but checkpoint step can be large).
    ema_smooth_steps = 0
    t_start_training = time.time()

    while True:
        torch.cuda.synchronize()
        t0 = time.time()
        inner = getattr(model, "_orig_mod", model)
        inner.kl_weight.fill_(_get_kl_weight(step))

        train_loss_accum = torch.zeros((), device=device)
        train_pc_accum = torch.zeros((), device=device)
        last_layer_means = None

        for _ in range(grad_accum_steps):
            with autocast_ctx:
                main_loss, aux_loss, layer_means = model(x, y)
                loss = main_loss + PC_WEIGHT * aux_loss
            train_loss_accum += main_loss.detach()
            train_pc_accum += aux_loss.detach()
            last_layer_means = layer_means
            (loss / grad_accum_steps).backward()
            x, y, epoch = next(train_loader)

        train_loss = (train_loss_accum / grad_accum_steps).item()
        train_pc = (train_pc_accum / grad_accum_steps).item()

        # Schedules (progress, lrm, muon_mom, muon_wd)
        progress = min(total_training_time / TIME_BUDGET, 1.0)
        lrm = _get_lr_multiplier(progress)
        muon_mom = (1 - min(step / 300, 1)) * 0.85 + min(step / 300, 1) * 0.95
        muon_wd = WEIGHT_DECAY * (1 - progress)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group.get("kind") == "muon":
                group["momentum"] = muon_mom
                group["weight_decay"] = muon_wd

        # Optimizer step (clip_grad, step, zero_grad)
        pre_clip_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        optimizer.step()
        model.zero_grad(set_to_none=True)

        # PC EMA update
        with torch.no_grad():
            inner.pc_ema.data.copy_(inner.pc_ema * PC_EMA_DECAY + last_layer_means * (1.0 - PC_EMA_DECAY))

        # Fail check
        if math.isnan(train_loss) or train_loss > 100:
            print(f"FAIL step {step} loss={train_loss:.4f}", flush=True)
            torch.save({"model": model.state_dict(), "config": asdict(config), "step": step}, "checkpoint_failed.pt")
            sys.exit(1)

        # Timing
        torch.cuda.synchronize()
        dt = time.time() - t0
        if step > 10:
            total_training_time += dt

        # Logging (debiased EMA; ema_smooth_steps so resume doesn't break the divisor)
        grad_norm = pre_clip_norm.item()
        smooth_train_loss = LOG_SMOOTH_BETA * smooth_train_loss + (1 - LOG_SMOOTH_BETA) * train_loss
        ema_smooth_steps += 1
        debiased = smooth_train_loss / (1 - LOG_SMOOTH_BETA ** ema_smooth_steps)
        tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
        mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / REFERENCE_BF16_PEAK_FLOPS
        kl_w = inner.kl_weight.item()
        print(f"\rstep {step:05d} ({100*progress:.1f}%) | loss: {debiased:.6f} | raw: {train_loss:.6f} | pc: {train_pc:.6f} | "
              f"kl_w: {kl_w:.4f} | gn: {grad_norm:.3f} | lrm: {lrm:.2f} | {dt*1000:.0f}ms | "
              f"{tok_per_sec:,} tok/s | mfu: {mfu:.1f}% | epoch: {epoch} | elapsed: {total_training_time:.0f}s    ",
              end="", flush=True)

        if PC_DIAG_INTERVAL > 0 and step > 0 and step % PC_DIAG_INTERVAL == 0:
            diag = inner.get_pc_diagnostics()
            pw = ",".join(f"{w:.2f}" for w in diag["pc_weights"])
            sig = ",".join(f"{s:.3f}" for s in diag["sigma_bias_exp"]) if diag["sigma_bias_exp"] else "—"
            print(f"\n  pc_diag (step {step}): pc_weights=[{pw}] sigma_bias=[{sig}]", flush=True)

        if VAL_INTERVAL > 0 and step > 0 and step % VAL_INTERVAL == 0:
            model.eval()
            with autocast_ctx, torch.no_grad():
                val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)
            model.train()
            print(f"\n  val_bpb (step {step}): {val_bpb:.6f}", flush=True)

        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif (step + 1) % 5000 == 0:
            gc.collect()

        step += 1
        if step > 10 and total_training_time >= TIME_BUDGET:
            break

    print()
    total_tokens = step * TOTAL_BATCH_SIZE

    model.eval()
    with autocast_ctx, torch.no_grad():
        val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

    t_end = time.time()
    startup = t_start_training - t_start
    steady_mfu = (100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10) / total_training_time / REFERENCE_BF16_PEAK_FLOPS
                 if total_training_time > 0 else 0)
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(config),
        "step": step,
        "total_training_time": total_training_time,
    }, "checkpoint.pt")

    print("---")
    print(f"val_bpb:          {val_bpb:.6f}")
    print(f"startup_seconds:  {startup:.1f} (tokenizer={t_tokenizer:.1f}s, init={t_model_init:.1f}s, compile={t_compile:.1f}s, prewarm={t_prewarm:.0f}s)")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"mfu_percent:      {steady_mfu:.2f}")
    print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:    {param_counts['total'] / 1e6:.1f}")
    print(f"depth:            {DEPTH}")


if __name__ == "__main__":
    main()
