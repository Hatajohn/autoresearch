# AutoResearch Session — March 15, 2026 (WSL, Session 1)

## Setup

- **Branch**: `autoresearch/mar15-wsl` (from `master`)
- **Dataset**: climbmix-400b-shuffle (internet-scale, 10 training shards + 1 val shard)
- **Platform**: WSL2 (Ubuntu 24.04), RTX 4080 16GB, Flash Attention 3 (kernels-community/flash-attn3), torch.compile enabled
- **Python**: 3.10.20 via uv, PyTorch 2.9.1+cu128
- **Prior context**: 26 experiments on Windows/TinyStories (sessions 1–2) — see sessions/session_mar15_1.md and session_mar15_2.md

## Platform Notes vs Windows

| Factor | Windows | WSL (this session) |
|---|---|---|
| Dataset | TinyStories (~22M tok/5min) | climbmix (internet-scale, ~500M tok/5min expected) |
| FA3 | SDPA fallback | kernels-community/flash-attn3 ✓ |
| torch.compile | Disabled | Enabled ✓ |
| MFU | ~2.4% | ~30-40% expected |
| Baseline val_bpb | 0.472842 | TBD (harder dataset, expect ~1.0) |
| Best Windows val_bpb | 0.451641 | TBD |

**First run cold start**: torch.compile + FA3 kernel compilation takes ~15 min on first run; cached thereafter.

**GPU cooldown**: 60s sleep + CUDA cache clear between every experiment run to prevent thermal throttling and memory fragmentation.

## Inherited Best Config (from Windows Session 2)

```python
TOTAL_BATCH_SIZE = 2**14      # Windows-tuned; need to re-explore for WSL throughput
DEVICE_BATCH_SIZE = 8         # Windows; WSL can likely use 128 (default)
DEPTH = 8
ASPECT_RATIO = 64             # 512-dim model
WEIGHT_DECAY = 0.0            # confirmed win: underfitting regime
MATRIX_LR = 0.04
WARMDOWN_RATIO = 0.7          # confirmed win
EMBEDDING_LR = 1.0            # confirmed win
UNEMBEDDING_LR = 0.010        # confirmed win: lm_head was severely lagging
ADAM_BETAS = (0.8, 0.95)
WINDOW_PATTERN = "SSSL"       # now viable with FA3 (Windows used "L" only)
```

## Experiment Results

| # | Commit | val_bpb | VRAM GB | Status | Description |
|---|---|---|---|---|---|
| baseline | — | — | — | pending | Stock train.py, climbmix, FA3+compile — establishing floor |

## Key Findings

*(to be filled as experiments complete)*

## Ideas Queue

**Phase 1 — Apply confirmed Windows wins:**
- [ ] Apply all 4 confirmed wins at once: WEIGHT_DECAY=0.0, WARMDOWN_RATIO=0.7, EMBEDDING_LR=1.0, UNEMBEDDING_LR=0.010

**Phase 2 — Throughput calibration:**
- [ ] TOTAL_BATCH_SIZE 2\*\*19→2\*\*18 (more steps, see if RTX 4080 benefits)
- [ ] TOTAL_BATCH_SIZE 2\*\*19→2\*\*17 (even more steps)

**Phase 3 — Architecture (new territory):**
- [ ] SwiGLU MLP (staged in Windows session 2, never run — `F.silu(gate)*up_proj`, same param count)
- [ ] WINDOW_PATTERN="SSSL" vs baseline (now viable with FA3)
- [ ] HEAD_DIM=64 (8 heads at 512-dim instead of 4)
- [ ] DEPTH=10 + ASPECT_RATIO=48 (deeper, same dim — OOM'd on Windows without compile)

**Phase 4 — Experimental:**
- [ ] Multi-token prediction (predict t+1 and t+2, double supervision density)
- [ ] SCALAR_LR tuning (x0/resid lambdas, currently 0.5)
- [ ] WARMUP_RATIO=0.05 (small warmup for harder dataset stability)
- [ ] MATRIX_LR=0.03 (may pair better with WARMDOWN_RATIO=0.7 on harder data)
