# AutoResearch Session — March 15, 2026

## Setup

- **Branch**: `autoresearch/mar15` (from `master` @ `5f5c6dd`)
- **Dataset**: TinyStories (`karpathy/tinystories-gpt4-clean`) — single 641MB parquet shard
- **Platform**: Windows, RTX GPU ~16GB VRAM, no Triton / no Flash Attention 3
- **Baseline config**: `DEPTH=8`, `DEVICE_BATCH_SIZE=8`, `TOTAL_BATCH_SIZE=2**14`, `WEIGHT_DECAY=0.0`, `MATRIX_LR=0.04`, `WARMDOWN_RATIO=0.5`

## Windows Port (pre-research, on `master`)

All changes were committed to `master` before the research branch was created:

| Change | Reason |
|---|---|
| `requires-python = ">=3.10, <3.14"` | `nvidia-cufile-cu12` doesn't resolve on Python 3.14+ |
| Removed `default = true` from PyTorch index | Was overriding PyPI, breaking `kernels` package install |
| FA3 wrapped in `try/except`, SDPA fallback | No Windows FA3 builds exist |
| Removed all 3 `@torch.compile` usages | Triton has no Windows support |
| `MuonAdamW` rewritten to plain Python scalars | 0-D CPU tensors in optimizer caused device mismatch without compile |
| `DEVICE_BATCH_SIZE = 4` | OOM without compile memory optimizations |
| `WINDOW_PATTERN = "L"` | Sliding window makes no sense without FA3 |
| `TOTAL_BATCH_SIZE = 2**16` | Reduced from `2**19` (H100 default) for Windows throughput |
| `torch.backends.cudnn.benchmark = True` | Small throughput gain |
| `torch.load(..., weights_only=True)` in `prepare.py` | Silence FutureWarning |

## Experiment Results

| # | Commit | val_bpb | VRAM GB | Status | Change |
|---|---|---|---|---|---|
| baseline | `5f5c6dd` | 0.472842 | 3.9 | **keep** | Starting point: Windows port on TinyStories |
| 1 | `049544b` | 0.469432 | 3.9 | **keep** | `TOTAL_BATCH_SIZE` 2\*\*16 → 2\*\*14 (4× more optimizer steps) |
| 2 | `363a859` | 0.489346 | 2.6 | discard | `DEPTH` 8 → 6 — too small even with 2493 steps |
| 3 | `8418b1a` | 0.485658 | 3.8 | discard | `TOTAL_BATCH_SIZE` 2\*\*14 → 2\*\*13 — too noisy (single tokens) |
| 4 | `ad03a07` | 0.469106 | 7.1 | **keep** | `DEVICE_BATCH_SIZE` 4 → 8 (single micro-step, slightly cleaner grads) |
| 5 | `dbb4159` | 0.478840 | 7.1 | discard | `WARMDOWN_RATIO` 0.5 → 0.3 — warmdown is important, hurt convergence |
| 6 | `1c016d9` | 0.474178 | 7.1 | discard | `MATRIX_LR` 0.04 → 0.05 — mildly too high |
| 7 | `122d714` | **0.455852** | 7.1 | **keep** | `WEIGHT_DECAY` 0.2 → 0.0 — **biggest win**, confirmed underfitting regime |
| 8 | `7c7ce21` | crash | 0.0 | crash | `DEPTH` 8 → 10 — OOM during eval + slower convergence |
| 9 | `4008453` | *pending* | — | — | GQA: `n_kv_head = n_head // 2` (cancelled, in progress) |

## Key Findings

1. **Underfitting, not overfitting**: Removing `WEIGHT_DECAY` gave the biggest improvement (-0.017 val_bpb). We're training on ~22M tokens with a 50M param model in 5 minutes — very light touch.
2. **Batch size sweet spot is ~16K tokens**: 2\*\*14 (16K) beats both 2\*\*16 (65K) and 2\*\*13 (8K). More frequent updates help but too small is noisy.
3. **Depth/width tradeoff**: DEPTH=8 beats both DEPTH=6 and DEPTH=10. Smaller = underpowered, bigger = too slow to converge + OOM risk.
4. **Warmdown matters**: 50% of time budget in cooldown is important. Reducing to 30% hurt.
5. **Throughput**: ~46K tok/sec, ~1390 steps in 5 minutes. Compare to H100 baseline: ~499M tokens vs our ~22M tokens.

## Current State

**Best result**: `val_bpb = 0.455852` (Exp 7, `122d714`)

**Active config** (on HEAD `4008453`, exp 9 pending run):
```python
TOTAL_BATCH_SIZE = 2**14
DEVICE_BATCH_SIZE = 8
DEPTH = 8
WEIGHT_DECAY = 0.0
MATRIX_LR = 0.04
WARMDOWN_RATIO = 0.5
n_kv_head = n_head // 2  # GQA, exp9 in progress
```

## Ideas Queue

- [ ] GQA with `n_kv_head = n_head // 2` (exp 9, in progress)
- [ ] Higher `EMBEDDING_LR` (currently 0.6 — given underfitting, try 1.0)
- [ ] `ASPECT_RATIO = 80` instead of 64 (wider model at same depth)
- [ ] Reduce `WARMUP_RATIO` experiments (currently 0.0 — already disabled)
- [ ] `ADAM_BETAS = (0.9, 0.95)` instead of `(0.8, 0.95)` (more momentum)
- [ ] Reduce `ns_steps` for Muon (5 → 3, faster orthogonalization, more wall-clock steps)
- [ ] `FINAL_LR_FRAC > 0` (e.g. 0.05) to prevent LR going to zero
