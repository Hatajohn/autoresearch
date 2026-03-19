"""
Autoresearch pretraining script. Single-GPU.
Usage: uv run train.py

Structure:
  - Env and imports (Flash Attention, prepare, model, optimizer)
  - Run constants (time budget, resume, val interval, checkpoint_steps) from env / --checkpoint-steps
  - Hyperparameters (architecture, optimization, PC)
  - RunConfig: normalized run-control settings (env defaults + CLI overrides)
  - main(run_cfg): signals → torch setup → tokenizer → config → model → optimizer →
    compile → prewarm → training loop → final eval & checkpoint
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.expanduser("~/.cache/torchinductor"))

import argparse
import gc
import math
import signal
import sys
import time
from dataclasses import asdict, dataclass

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

TIME_BUDGET = int(os.environ.get("TRAIN_TIME_BUDGET", "86400"))
RESUME_CHECKPOINT = os.environ.get("RESUME_CHECKPOINT", "")
VAL_INTERVAL = int(os.environ.get("VAL_INTERVAL", "0"))
# Periodic recovery checkpoint (checkpoint_latest.pt). Override with --checkpoint-steps.
CHECKPOINT_STEPS_DEFAULT = int(os.environ.get("CHECKPOINT_STEPS", "300"))
MAX_EPOCHS_DEFAULT = int(os.environ.get("MAX_EPOCHS", "1"))
# Automatic save-and-stop policy. Validation patience is the primary stop signal;
# raw train loss is a safety stop for clearly bad runs before more time is wasted.
EARLY_STOP_ENABLE_DEFAULT = os.environ.get("EARLY_STOP_ENABLE", "1").strip().lower() not in ("0", "false", "off", "no")
EARLY_STOP_MIN_STEPS_DEFAULT = int(os.environ.get("EARLY_STOP_MIN_STEPS", "50"))
EARLY_STOP_VAL_PATIENCE_DEFAULT = int(os.environ.get("EARLY_STOP_VAL_PATIENCE", "3"))
EARLY_STOP_VAL_MIN_DELTA_DEFAULT = float(os.environ.get("EARLY_STOP_VAL_MIN_DELTA", "0.005"))
EARLY_STOP_TRAIN_PATIENCE_DEFAULT = int(os.environ.get("EARLY_STOP_TRAIN_PATIENCE", "40"))
EARLY_STOP_TRAIN_MIN_DELTA_DEFAULT = float(os.environ.get("EARLY_STOP_TRAIN_MIN_DELTA", "0.02"))

# Initial steps excluded from throughput/MFU reporting (JIT-warmup outliers).
# The LR/WD schedule clock and time-budget stop are NOT affected — both track from step 0.
MFU_WARMUP_STEPS = 10
# Muon momentum ramps from 0.85→0.95 over this many completed optimizer steps.
# On resume, continue from the restored step so the coefficient matches the
# warm optimizer buffers loaded from checkpoint.
MUON_MOM_WARMUP_STEPS = 300
# Explicit GC pass every N steps (keeps long-lived tensor refs from accumulating).
GC_INTERVAL = 5000

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

# Architecture
DEPTH = int(os.environ.get("DEPTH", "8"))
ASPECT_RATIO = int(os.environ.get("ASPECT_RATIO", "32"))
MODEL_WIDTH_OVERRIDE = os.environ.get("MODEL_WIDTH", os.environ.get("WIDTH"))
HEAD_DIM = 64
WINDOW_PATTERN = os.environ.get("WINDOW_PATTERN", "LOG").strip().upper() or "LOG"
DEVICE_BATCH_SIZE = 32
if DEPTH <= 0:
    raise ValueError(f"DEPTH must be positive, got {DEPTH}")
if ASPECT_RATIO <= 0:
    raise ValueError(f"ASPECT_RATIO must be positive, got {ASPECT_RATIO}")
if MODEL_WIDTH_OVERRIDE is None:
    TARGET_MODEL_WIDTH = DEPTH * ASPECT_RATIO
    WIDTH_MODE = f"aspect_ratio({ASPECT_RATIO})"
else:
    TARGET_MODEL_WIDTH = int(MODEL_WIDTH_OVERRIDE)
    if TARGET_MODEL_WIDTH <= 0:
        raise ValueError(f"MODEL_WIDTH must be positive, got {TARGET_MODEL_WIDTH}")
    WIDTH_MODE = "model_width"
PC_HEAD_DIM = int(os.environ.get("PC_HEAD_DIM", str(max(64, TARGET_MODEL_WIDTH // DEPTH))))
LOG_MIN_WINDOW = int(os.environ.get("LOG_MIN_WINDOW", "0"))

# Optimization
TOTAL_BATCH_SIZE = 2**18
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


# ---------------------------------------------------------------------------
# RunConfig: normalized run-control settings
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    """
    Single source of truth for run-control settings.
    Build once from env defaults + CLI overrides; pass to main().
    Keeps precedence logic local to __main__ and out of the training loop.
    """
    checkpoint_steps: int = CHECKPOINT_STEPS_DEFAULT
    max_epochs: int = MAX_EPOCHS_DEFAULT
    early_stop_enable: bool = EARLY_STOP_ENABLE_DEFAULT
    early_stop_min_steps: int = EARLY_STOP_MIN_STEPS_DEFAULT
    early_stop_val_patience: int = EARLY_STOP_VAL_PATIENCE_DEFAULT
    early_stop_val_min_delta: float = EARLY_STOP_VAL_MIN_DELTA_DEFAULT
    early_stop_train_patience: int = EARLY_STOP_TRAIN_PATIENCE_DEFAULT
    early_stop_train_min_delta: float = EARLY_STOP_TRAIN_MIN_DELTA_DEFAULT


_PENDING_STOP_REASON = None


def _request_stop(reason: str) -> None:
    global _PENDING_STOP_REASON
    if _PENDING_STOP_REASON is None:
        _PENDING_STOP_REASON = reason


def _consume_stop_request() -> str | None:
    global _PENDING_STOP_REASON
    reason = _PENDING_STOP_REASON
    _PENDING_STOP_REASON = None
    return reason


def _setup_signals():
    def _handle(signum, frame):
        name = getattr(signal.Signals(signum), "name", signum)
        _request_stop(f"received {name}")
        print(f"\n[train] Received {name} — will checkpoint and stop at the next safe boundary.", flush=True)
    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)


def _build_config(vocab_size):
    base_dim = TARGET_MODEL_WIDTH
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


def _rebind_optimizer_to_model(model, optimizer):
    """Point optimizer groups and state at the live model parameters."""
    inner = getattr(model, "_orig_mod", model)
    if not hasattr(inner, "_get_optimizer_param_lists"):
        return
    param_lists = inner._get_optimizer_param_lists()
    assert len(param_lists) == len(optimizer.param_groups)
    old_flat = [p for g in optimizer.param_groups for p in g["params"]]
    new_flat = [p for lst in param_lists for p in lst]
    assert len(old_flat) == len(new_flat)
    for old_p, new_p in zip(old_flat, new_flat):
        if old_p is new_p:
            continue
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


def _validate_startup_config(run_cfg: RunConfig) -> None:
    if TIME_BUDGET <= 0:
        raise ValueError(f"TRAIN_TIME_BUDGET must be positive, got {TIME_BUDGET}")
    if KL_WARMUP_STEPS <= 0:
        raise ValueError(f"KL_WARMUP_STEPS must be positive, got {KL_WARMUP_STEPS}")
    if VAL_INTERVAL < 0:
        raise ValueError(f"VAL_INTERVAL must be non-negative, got {VAL_INTERVAL}")
    if run_cfg.checkpoint_steps < 0:
        raise ValueError(f"CHECKPOINT_STEPS must be non-negative, got {run_cfg.checkpoint_steps}")
    if run_cfg.max_epochs < 0:
        raise ValueError(f"MAX_EPOCHS must be non-negative, got {run_cfg.max_epochs}")
    if run_cfg.early_stop_min_steps < 0:
        raise ValueError(f"EARLY_STOP_MIN_STEPS must be non-negative, got {run_cfg.early_stop_min_steps}")
    if run_cfg.early_stop_val_patience < 0:
        raise ValueError(f"EARLY_STOP_VAL_PATIENCE must be non-negative, got {run_cfg.early_stop_val_patience}")
    if run_cfg.early_stop_train_patience < 0:
        raise ValueError(f"EARLY_STOP_TRAIN_PATIENCE must be non-negative, got {run_cfg.early_stop_train_patience}")
    if run_cfg.early_stop_val_min_delta < 0:
        raise ValueError(f"EARLY_STOP_VAL_MIN_DELTA must be non-negative, got {run_cfg.early_stop_val_min_delta}")
    if run_cfg.early_stop_train_min_delta < 0:
        raise ValueError(f"EARLY_STOP_TRAIN_MIN_DELTA must be non-negative, got {run_cfg.early_stop_train_min_delta}")


def _get_kl_weight(step):
    if KL_WARMUP_STEPS <= 0:
        raise ValueError(f"KL_WARMUP_STEPS must be positive, got {KL_WARMUP_STEPS}")
    return KL_WEIGHT * min(step / KL_WARMUP_STEPS, 1.0)


def _should_evaluate(step: int, val_interval: int) -> bool:
    return val_interval > 0 and step > 0 and step % val_interval == 0


def _should_stop_for_epoch_limit(next_epoch: int, max_epochs: int) -> bool:
    return max_epochs > 0 and next_epoch > max_epochs


def _format_hms(seconds):
    """e.g. 03::12::05 (11285 seconds) — HH::MM::SS then total seconds in parentheses."""
    t = max(0, int(round(float(seconds))))
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}::{m:02d}::{s:02d} ({t} seconds)"


def _save_recovery_checkpoint(model, optimizer, config, step, schedule_time, run_state=None, path="checkpoint_latest.pt"):
    if run_state is None:
        run_state = {
            "step": step,
            "schedule_time": schedule_time,
        }
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(config),
        "step": step,
        "total_training_time": schedule_time,
        "run_state": run_state,
    }, path)


def _load_run_state(ckpt: dict) -> dict:
    run_state = dict(ckpt.get("run_state", {}))
    if "step" not in run_state:
        run_state["step"] = ckpt.get("step", 0)
    if "schedule_time" not in run_state:
        run_state["schedule_time"] = ckpt.get("total_training_time", 0.0)
    return run_state


def _normalize_state_dict_keys(state_dict: dict) -> dict:
    """
    Support eager checkpoints plus both whole-model and regional torch.compile
    wrappers, which can introduce `_orig_mod` at different path segments.
    """
    normalized = {}
    for key, value in state_dict.items():
        key = key.removeprefix("_orig_mod.")
        key = key.replace("._orig_mod.", ".")
        normalized[key] = value
    return normalized


def _current_run_state(
    *,
    step: int,
    schedule_time: float,
    smooth_train_loss: float,
    ema_smooth_steps: int,
    best_val_bpb: float,
    val_bad_checks: int,
    best_raw_train_loss: float,
    train_bad_steps: int,
    throughput_time: float,
) -> dict:
    return {
        "step": step,
        "schedule_time": schedule_time,
        "smooth_train_loss": smooth_train_loss,
        "ema_smooth_steps": ema_smooth_steps,
        "best_val_bpb": best_val_bpb,
        "val_bad_checks": val_bad_checks,
        "best_raw_train_loss": best_raw_train_loss,
        "train_bad_steps": train_bad_steps,
        "throughput_time": throughput_time,
    }


def main(run_cfg: "RunConfig | None" = None):
    if run_cfg is None:
        run_cfg = RunConfig()
    _validate_startup_config(run_cfg)

    _consume_stop_request()
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

    # Resume policy (authoritative for all mutable state):
    #   model/optimizer:   restored — preserve parameter values and warm optimizer buffers
    #   step:              restored — preserves checkpoint numbering and schedule position
    #   schedule_time:     restored — crash recovery continues the interrupted run budget
    #   pc_ema:            restored via model.state_dict() (persistent buffer)
    #   smoothing/early-stop state: restored so resumed runs behave like uninterrupted ones
    #   muon_warmup_step:  set to restored step — matches the warm optimizer buffers
    #   kl_weight:         uses restored step via _get_kl_weight(step) — continuation semantics
    step = 0
    schedule_time = 0.0
    smooth_train_loss = 0.0
    ema_smooth_steps = 0
    best_val_bpb = float("inf")
    val_bad_checks = 0
    best_raw_train_loss = float("inf")
    train_bad_steps = 0
    throughput_time = 0.0
    stop_reason = None
    if RESUME_CHECKPOINT:
        ckpt = torch.load(RESUME_CHECKPOINT, map_location=device, weights_only=True)
        state = _normalize_state_dict_keys(ckpt["model"])
        model.load_state_dict(state, assign=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        _rebind_optimizer_to_model(model, optimizer)
        run_state = _load_run_state(ckpt)
        step = int(run_state.get("step", 0))
        schedule_time = float(run_state.get("schedule_time", 0.0))
        smooth_train_loss = float(run_state.get("smooth_train_loss", 0.0))
        ema_smooth_steps = int(run_state.get("ema_smooth_steps", 0))
        best_val_bpb = float(run_state.get("best_val_bpb", float("inf")))
        val_bad_checks = int(run_state.get("val_bad_checks", 0))
        best_raw_train_loss = float(run_state.get("best_raw_train_loss", float("inf")))
        train_bad_steps = int(run_state.get("train_bad_steps", 0))
        throughput_time = float(run_state.get("throughput_time", 0.0))
        print(
            f"Resumed from {RESUME_CHECKPOINT} at step {step} "
            f"(schedule_time={schedule_time:.1f}s, val_bad_checks={val_bad_checks}, train_bad_steps={train_bad_steps})",
            flush=True,
        )

    t0 = time.time()
    if TORCH_COMPILE_MODE == "full":
        print("Compiling full model (torch.compile)...", flush=True)
        model = torch.compile(model, dynamic=False, fullgraph=True)
        t_compile = time.time() - t0
        print(f"  torch.compile done ({t_compile:.1f}s)", flush=True)
        _rebind_optimizer_to_model(model, optimizer)
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
        _rebind_optimizer_to_model(model, optimizer)
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
    if run_cfg.checkpoint_steps > 0:
        print(f"Periodic checkpoint every {run_cfg.checkpoint_steps} steps → checkpoint_latest.pt", flush=True)
    if run_cfg.max_epochs > 0:
        print(f"Max epochs: {run_cfg.max_epochs} full shard pass(es)", flush=True)
    if run_cfg.early_stop_enable:
        print(
            "Early stop enabled: "
            f"min_steps={run_cfg.early_stop_min_steps}, "
            f"val_patience={run_cfg.early_stop_val_patience}, val_min_delta={run_cfg.early_stop_val_min_delta}, "
            f"train_patience={run_cfg.early_stop_train_patience}, train_min_delta={run_cfg.early_stop_train_min_delta}",
            flush=True,
        )
        if VAL_INTERVAL <= 0:
            print("  VAL_INTERVAL=0, so validation early stop is inactive; raw-loss safety stop remains active.", flush=True)
    print("Starting training loop...", flush=True)

    all_params = [p for g in optimizer.param_groups for p in g["params"]]
    # Debiased EMA uses its own step counter so resumed runs keep the same smoothing state.
    t_start_training = time.time()
    # Throughput time for MFU reporting only; excludes MFU_WARMUP_STEPS initial steps.
    # It is restored on resume so steady-MFU reporting stays continuous.
    # Muon momentum coefficient warmup counter. Initialised to `step` (not 0) so it
    # matches the restored Muon optimizer buffers: fresh run → step=0 → warmup starts
    # at 0.85; resumed run → step=N → coefficient starts at wherever it was, consistent
    # with the warm momentum/second-momentum buffers from optimizer.load_state_dict().
    muon_warmup_step = step

    while True:
        pending_stop = _consume_stop_request()
        if pending_stop is not None:
            stop_reason = pending_stop
            _save_recovery_checkpoint(
                model,
                optimizer,
                config,
                step,
                schedule_time,
                run_state=_current_run_state(
                    step=step,
                    schedule_time=schedule_time,
                    smooth_train_loss=smooth_train_loss,
                    ema_smooth_steps=ema_smooth_steps,
                    best_val_bpb=best_val_bpb,
                    val_bad_checks=val_bad_checks,
                    best_raw_train_loss=best_raw_train_loss,
                    train_bad_steps=train_bad_steps,
                    throughput_time=throughput_time,
                ),
                path="checkpoint_latest.pt",
            )
            print(f"  saved checkpoint_latest.pt (step {step}) — {stop_reason}", flush=True)
            break

        torch.cuda.synchronize()
        t0 = time.time()
        inner = getattr(model, "_orig_mod", model)
        inner.kl_weight.fill_(_get_kl_weight(step))

        train_loss_accum = torch.zeros((), device=device)
        train_pc_accum = torch.zeros((), device=device)
        layer_means_accum = None

        for _ in range(grad_accum_steps):
            with autocast_ctx:
                main_loss, aux_loss, layer_means = model(x, y)
                loss = main_loss + PC_WEIGHT * aux_loss
            train_loss_accum += main_loss.detach()
            train_pc_accum += aux_loss.detach()
            # Accumulate detached layer_means across all micro-batches so the
            # EMA update reflects the full effective batch, not just the last micro-batch.
            if layer_means_accum is None:
                layer_means_accum = layer_means.detach()
            else:
                layer_means_accum = layer_means_accum + layer_means.detach()
            (loss / grad_accum_steps).backward()
            x, y, epoch = next(train_loader)

        train_loss = (train_loss_accum / grad_accum_steps).item()
        train_pc = (train_pc_accum / grad_accum_steps).item()
        avg_layer_means = layer_means_accum / grad_accum_steps

        # Schedules (progress, lrm, muon_mom, muon_wd)
        progress = min(schedule_time / TIME_BUDGET, 1.0)
        lrm = _get_lr_multiplier(progress)
        t_muon = min(muon_warmup_step / MUON_MOM_WARMUP_STEPS, 1.0)
        muon_mom = (1 - t_muon) * 0.85 + t_muon * 0.95
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

        # PC EMA update: use avg_layer_means (full-batch average, not last micro-batch)
        with torch.no_grad():
            inner.pc_ema.data.copy_(inner.pc_ema * PC_EMA_DECAY + avg_layer_means * (1.0 - PC_EMA_DECAY))

        # Treat `step` as completed optimizer updates from here onward.
        step += 1
        muon_warmup_step += 1

        # Fail check
        if math.isnan(train_loss) or train_loss > 100:
            print(f"FAIL step {step} loss={train_loss:.4f}", flush=True)
            torch.save({"model": model.state_dict(), "config": asdict(config), "step": step}, "checkpoint_failed.pt")
            sys.exit(1)

        # Timing: schedule_time drives LR/WD schedules and the total run time budget.
        # throughput_time drives steady-MFU reporting only (excludes MFU_WARMUP_STEPS).
        torch.cuda.synchronize()
        dt = time.time() - t0
        schedule_time += dt
        if step > MFU_WARMUP_STEPS:
            throughput_time += dt

        # Logging (debiased EMA; ema_smooth_steps so resume doesn't break the divisor)
        grad_norm = pre_clip_norm.item()
        smooth_train_loss = LOG_SMOOTH_BETA * smooth_train_loss + (1 - LOG_SMOOTH_BETA) * train_loss
        ema_smooth_steps += 1
        debiased = smooth_train_loss / (1 - LOG_SMOOTH_BETA ** ema_smooth_steps)
        tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
        mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / REFERENCE_BF16_PEAK_FLOPS
        kl_w = inner.kl_weight.item()
        print(f"step {step:05d} ({100*progress:.1f}%) | loss: {debiased:.6f} | raw: {train_loss:.6f} | pc: {train_pc:.6f} | "
              f"kl_w: {kl_w:.4f} | gn: {grad_norm:.3f} | lrm: {lrm:.2f} | {dt*1000:.0f}ms | "
              f"{tok_per_sec:,} tok/s | mfu: {mfu:.1f}% | epoch: {epoch} | elapsed: {schedule_time:.0f}s",
              flush=True)
        should_stop = False
        pending_stop = _consume_stop_request()
        signal_stop = pending_stop is not None
        if signal_stop:
            should_stop = True
            stop_reason = pending_stop

        if PC_DIAG_INTERVAL > 0 and step > 0 and step % PC_DIAG_INTERVAL == 0:
            diag = inner.get_pc_diagnostics()
            pw = ",".join(f"{w:.2f}" for w in diag["pc_weights"])
            sig = ",".join(f"{s:.3f}" for s in diag["sigma_bias_exp"]) if diag["sigma_bias_exp"] else "—"
            print(f"  pc_diag (step {step}): pc_weights=[{pw}] sigma_bias=[{sig}]", flush=True)

        if not signal_stop and _should_evaluate(step, VAL_INTERVAL):
            model.eval()
            with autocast_ctx, torch.no_grad():
                val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)
            model.train()
            print(f"  val_bpb (step {step}): {val_bpb:.6f}", flush=True)
            if run_cfg.early_stop_enable and step >= run_cfg.early_stop_min_steps and run_cfg.early_stop_val_patience > 0:
                if val_bpb < best_val_bpb - run_cfg.early_stop_val_min_delta:
                    best_val_bpb = val_bpb
                    val_bad_checks = 0
                else:
                    val_bad_checks += 1
                    if val_bad_checks >= run_cfg.early_stop_val_patience:
                        should_stop = True
                        stop_reason = (
                            f"early stop: val_bpb {val_bpb:.6f} failed to beat best {best_val_bpb:.6f} "
                            f"by {run_cfg.early_stop_val_min_delta:.6f} for {val_bad_checks} validation checks"
                        )

        if train_loss < best_raw_train_loss:
            best_raw_train_loss = train_loss
        if (
            run_cfg.early_stop_enable
            and not should_stop
            and step >= run_cfg.early_stop_min_steps
            and run_cfg.early_stop_train_patience > 0
        ):
            if train_loss <= best_raw_train_loss + run_cfg.early_stop_train_min_delta:
                train_bad_steps = 0
            else:
                train_bad_steps += 1
                if train_bad_steps >= run_cfg.early_stop_train_patience:
                    should_stop = True
                    stop_reason = (
                        f"early stop: raw loss {train_loss:.6f} stayed above best {best_raw_train_loss:.6f} "
                        f"+ {run_cfg.early_stop_train_min_delta:.6f} for {train_bad_steps} steps"
                    )

        if not should_stop and _should_stop_for_epoch_limit(epoch, run_cfg.max_epochs):
            should_stop = True
            stop_reason = (
                f"completed max_epochs={run_cfg.max_epochs} "
                f"(next batch would enter epoch {epoch})"
            )

        save_recovery = (run_cfg.checkpoint_steps > 0 and step > 0 and step % run_cfg.checkpoint_steps == 0) or should_stop
        if save_recovery:
            _save_recovery_checkpoint(
                model,
                optimizer,
                config,
                step,
                schedule_time,
                run_state=_current_run_state(
                    step=step,
                    schedule_time=schedule_time,
                    smooth_train_loss=smooth_train_loss,
                    ema_smooth_steps=ema_smooth_steps,
                    best_val_bpb=best_val_bpb,
                    val_bad_checks=val_bad_checks,
                    best_raw_train_loss=best_raw_train_loss,
                    train_bad_steps=train_bad_steps,
                    throughput_time=throughput_time,
                ),
                path="checkpoint_latest.pt",
            )
            msg = f"  saved checkpoint_latest.pt (step {step})"
            if should_stop:
                msg += f" — {stop_reason}"
            print(msg, flush=True)

        if step == 1:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif step % GC_INTERVAL == 0:
            gc.collect()
        if should_stop:
            break
        if schedule_time >= TIME_BUDGET:
            break

    total_tokens = step * TOTAL_BATCH_SIZE

    model.eval()
    with autocast_ctx, torch.no_grad():
        val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

    t_end = time.time()
    startup = t_start_training - t_start
    mfu_counted_steps = max(step - MFU_WARMUP_STEPS, 0)
    steady_mfu = (
        100 * num_flops_per_token * TOTAL_BATCH_SIZE * mfu_counted_steps / throughput_time / REFERENCE_BF16_PEAK_FLOPS
        if throughput_time > 0 else 0
    )
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(config),
        "step": step,
        "total_training_time": schedule_time,
        "run_state": _current_run_state(
            step=step,
            schedule_time=schedule_time,
            smooth_train_loss=smooth_train_loss,
            ema_smooth_steps=ema_smooth_steps,
            best_val_bpb=best_val_bpb,
            val_bad_checks=val_bad_checks,
            best_raw_train_loss=best_raw_train_loss,
            train_bad_steps=train_bad_steps,
            throughput_time=throughput_time,
        ),
    }, "checkpoint.pt")

    print("---")
    print(f"val_bpb:          {val_bpb:.6f}")
    if stop_reason is not None:
        print(f"stop_reason:      {stop_reason}")
    print(f"startup_seconds:  {_format_hms(startup)}  tokenizer={t_tokenizer:.1f}s, init={t_model_init:.1f}s, compile={t_compile:.1f}s, prewarm={t_prewarm:.0f}s")
    print(f"training_seconds: {_format_hms(schedule_time)}")
    print(f"total_seconds:    {_format_hms(t_end - t_start)}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"mfu_percent:      {steady_mfu:.2f}")
    print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:    {param_counts['total'] / 1e6:.1f}")
    print(f"depth:            {DEPTH}")
    print(f"target_width:     {TARGET_MODEL_WIDTH}")
    print(f"width_mode:       {WIDTH_MODE}")
    print(f"window_pattern:   {WINDOW_PATTERN}")


if __name__ == "__main__":
    _ap = argparse.ArgumentParser(description="Autoresearch pretraining")
    _ap.add_argument(
        "--checkpoint-steps",
        type=int,
        default=None,
        metavar="N",
        help="Save checkpoint_latest.pt every N steps (0=off). Overrides CHECKPOINT_STEPS env; default 300.",
    )
    _ap.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N full passes through the training shards (0=off). Overrides MAX_EPOCHS.",
    )
    _ap.add_argument(
        "--early-stop-enable",
        type=int,
        choices=(0, 1),
        default=None,
        metavar="0|1",
        help="Enable automatic save-and-stop (1=on, 0=off). Overrides EARLY_STOP_ENABLE.",
    )
    _ap.add_argument("--early-stop-min-steps", type=int, default=None, metavar="N", help="Do not early-stop before N steps.")
    _ap.add_argument(
        "--early-stop-val-patience",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N non-improving validation checks. Overrides EARLY_STOP_VAL_PATIENCE.",
    )
    _ap.add_argument(
        "--early-stop-val-min-delta",
        type=float,
        default=None,
        metavar="X",
        help="Required val_bpb improvement to reset validation patience. Overrides EARLY_STOP_VAL_MIN_DELTA.",
    )
    _ap.add_argument(
        "--early-stop-train-patience",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N raw-loss safety misses. Overrides EARLY_STOP_TRAIN_PATIENCE.",
    )
    _ap.add_argument(
        "--early-stop-train-min-delta",
        type=float,
        default=None,
        metavar="X",
        help="Allowed raw-loss drift above best before counting as a miss. Overrides EARLY_STOP_TRAIN_MIN_DELTA.",
    )
    _cli = _ap.parse_args()
    _run_cfg = RunConfig(
        checkpoint_steps=(_cli.checkpoint_steps if _cli.checkpoint_steps is not None else CHECKPOINT_STEPS_DEFAULT),
        max_epochs=(_cli.max_epochs if _cli.max_epochs is not None else MAX_EPOCHS_DEFAULT),
        early_stop_enable=(bool(_cli.early_stop_enable) if _cli.early_stop_enable is not None else EARLY_STOP_ENABLE_DEFAULT),
        early_stop_min_steps=(_cli.early_stop_min_steps if _cli.early_stop_min_steps is not None else EARLY_STOP_MIN_STEPS_DEFAULT),
        early_stop_val_patience=(_cli.early_stop_val_patience if _cli.early_stop_val_patience is not None else EARLY_STOP_VAL_PATIENCE_DEFAULT),
        early_stop_val_min_delta=(_cli.early_stop_val_min_delta if _cli.early_stop_val_min_delta is not None else EARLY_STOP_VAL_MIN_DELTA_DEFAULT),
        early_stop_train_patience=(_cli.early_stop_train_patience if _cli.early_stop_train_patience is not None else EARLY_STOP_TRAIN_PATIENCE_DEFAULT),
        early_stop_train_min_delta=(_cli.early_stop_train_min_delta if _cli.early_stop_train_min_delta is not None else EARLY_STOP_TRAIN_MIN_DELTA_DEFAULT),
    )
    main(_run_cfg)
