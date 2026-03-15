# AutoResearch Session — March 15, 2026 (WSL, Session 1)

## Setup

- **Branch**: `autoresearch/mar15-wsl` (from `master` @ `2c82e32`)
- **Dataset**: climbmix-400b-shuffle (internet-scale, 10 training shards + 1 val shard pinned at shard_06542)
- **Platform**: WSL2 (Ubuntu 24.04), RTX 4080 16GB, Flash Attention 3 (kernels-community/flash-attn3), torch.compile enabled
- **Python**: 3.10.20 via uv, PyTorch 2.9.1+cu128, gcc 13.3
- **Prior context**: 26 experiments on Windows/TinyStories — see sessions/session_mar15_1.md and sessions/session_mar15_2.md

## Platform Profile vs Windows

| Factor | Windows | WSL (this session) |
|---|---|---|
| Dataset | TinyStories (~22M tok/5min) | climbmix (internet-scale) |
| FA3 | SDPA fallback | kernels-community/flash-attn3 |
| torch.compile | Disabled | Enabled |
| MFU (vs H100 ref) | ~2.4% | ~6.5% (~75% of RTX 4080 actual peak) |
| tok/sec | ~87K | ~260K |
| Baseline val_bpb | 0.472842 (TinyStories) | **1.186619** (climbmix) |
| Best val_bpb | 0.451641 | **1.186619** (so far) |
| VRAM at DEVICE_BS=8 | ~7.1 GB | **3.3 GB** (16GB available — big headroom) |
| Steps/5min | ~1,400 | **164** (bottleneck) |
| Tokens/5min | ~22M | **86M** |

## WSL Port Changes vs upstream train.py (commits 37f9b93, 2c82e32)

| Change | Reason |
|---|---|
| FA3 wrapped in `try/except` with SDPA fallback | Resilience; mirrors Windows port |
| `PYTHONUNBUFFERED=1` + `flush=True` on early prints | run.log gets output even if OOM-killed before first print |
| `torch.backends.cudnn.benchmark = True` | Small throughput gain |
| `DEVICE_BATCH_SIZE = 8` (was 128) | RTX 4080 OOMed at 128; 3.3GB VRAM at 8 |
| `cooldown.py` added | 60s GPU cooldown + CUDA cache clear between runs |
| `sessions/` directory added | Session documentation |

## Experiment Results

| # | Commit | val_bpb | VRAM GB | Status | Description |
|---|---|---|---|---|---|
| baseline | 2c82e32 | 1.186619 | 3.3 | **keep** | Stock defaults: WEIGHT_DECAY=0.2, WARMDOWN=0.5, EMBED_LR=0.6, UNEMBED_LR=0.004, DEVICE_BS=8, TOTAL_BS=2**19 — 164 steps/86M tok |
| 1 | 66b49f1 | 1.207893 | 3.3 | discard | All 4 Windows LR wins combined — hurt because WARMDOWN=0.7 leaves only 49 of 164 steps at full LR (too aggressive) |

## Key Findings

1. **Step count is the binding constraint on this platform.** 164 optimizer steps in 5 minutes vs 1,400 on Windows. Root cause: DEVICE_BATCH_SIZE=8 → grad_accum_steps=32, meaning 32 sequential micro-batches before each optimizer update. The GPU is underutilized: VRAM is only 3.3 GB out of 16 GB.

2. **Windows LR hyperparameters don't blindly transfer.** WARMDOWN_RATIO=0.7 was a win at 1,400 steps (980 steps at full LR, 420 decay). At 164 steps it leaves only ~49 steps at full LR — clearly too aggressive. All 4 Windows wins applied together made things **worse** by 0.021 bpb.

3. **Throughput fix is priority 1.** Increasing DEVICE_BATCH_SIZE from 8 to 32–64 will reduce grad_accum_steps from 32 to 4–8. This should give ~400+ steps per 5 min. VRAM headroom is ample (3.3 GB used of 16 GB).

4. **FA3 loads correctly on RTX 4080** (compute cap 8.9, `kernels-community/flash-attn3`, 768 MB kernel). No issues.

5. **climbmix is a much harder dataset** than TinyStories. Baseline val_bpb = 1.186 vs 0.472 on TinyStories. The underfitting hypothesis from Windows may not hold — climbmix has far more entropy.

6. **First run cold start**: torch.compile + FA3 compilation takes ~20 min cold but is cached. All subsequent runs start in ~20s.

## Next Experiments (Priority Order)

- [ ] **Exp 2**: `DEVICE_BATCH_SIZE` 8→64 (grad_accum 32→4, expect ~400+ steps, proper GPU utilization)
- [ ] **Exp 3**: If exp 2 OOMs, try 32 instead
- [ ] **Exp 4**: `TOTAL_BATCH_SIZE` 2\*\*19→2\*\*17 (more optimizer steps at same micro-batch size)
- [ ] **Exp 5**: `UNEMBEDDING_LR` 0.004→0.010 alone (biggest single Windows win — likely transfers)
- [ ] **Exp 6**: `WEIGHT_DECAY` 0.2→0.0 alone (check if climbmix is also underfitting)
- [ ] **Exp 7**: `WARMDOWN_RATIO` calibrated to new step count (after throughput fix)
- [ ] **Exp 8**: SwiGLU MLP (fully implemented in Windows train.py, never evaluated)
- [ ] **Exp 9**: `HEAD_DIM=64` (8 heads at 512-dim instead of 4 — richer attention)

## Longer-Term Ideas

- Multi-token prediction (predict t+1 and t+2 simultaneously, double supervision signal per batch)
- SCALAR_LR tuning (x0_lambdas/resid_lambdas currently 0.5)
- WARMUP_RATIO=0.05 (may help stability on harder dataset)
- DEPTH=10 + ASPECT_RATIO=48 (deeper same-dim model — was OOM on Windows, should work with compile)
- WINDOW_PATTERN="SSSL" (now viable with FA3; Windows had to use "L")
- MATRIX_LR=0.03 (slight LR reduction to pair with longer warmdown once calibrated)
