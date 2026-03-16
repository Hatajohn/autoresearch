#!/usr/bin/env python3
"""
Smoke test for train.py.  Two tests, increasing in fidelity:

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

Usage:
  python smoke_test.py              # run both tests
  python smoke_test.py --no-compile # skip Test 2 (faster, e.g. in CI)
"""

import math
import os
import sys

import torch

# Save flags before patching sys.argv (train.py's module-level code must not
# see unknown args, but we still need our own flags after import).
_smoke_args = sys.argv[1:]
sys.argv = [sys.argv[0]]

# train.py reads env-vars at module level; supply sane defaults if not set
os.environ.setdefault("PC_WEIGHT",      "0.1")
os.environ.setdefault("PC_ALPHA",       "0.1")
os.environ.setdefault("PC_FOCAL_GAMMA", "1.0")
os.environ.setdefault("KL_WEIGHT",      "0.01")

sys.path.insert(0, os.path.dirname(__file__))
from train import GPT, GPTConfig  # noqa: E402

DEVICE   = torch.device("cuda")
AUTOCAST = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def assert_finite(tensor, name: str):
    if tensor.isnan().any():
        raise AssertionError(f"{name} contains NaN")
    if tensor.isinf().any():
        raise AssertionError(f"{name} contains Inf")


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
    pc_alpha  = model.pc_alpha.item()
    assert pc_alpha >= 0, f"pc_alpha negative ({pc_alpha})"

    x = torch.randint(0, config.vocab_size, (1, 256), device=DEVICE)
    y = torch.randint(0, config.vocab_size, (1, 256), device=DEVICE)

    with AUTOCAST:
        main_loss, aux_loss = model(x, y)
        (main_loss + 0.1 * aux_loss).backward()

    assert_finite(main_loss, "main_loss")
    assert_finite(aux_loss,  "aux_loss")
    assert math.isfinite(main_loss.item()), f"main_loss not finite: {main_loss.item()}"

    print(f"{PASS}  loss={main_loss.item():.4f}  pc_scale={pc_scale:.2f}")


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
        main_loss, aux_loss = model(x, y)
        (main_loss + 0.1 * aux_loss).backward()

    assert_finite(main_loss, "main_loss (compiled)")
    assert math.isfinite(main_loss.item()), f"compiled main_loss not finite: {main_loss.item()}"

    print(f"{PASS}  loss={main_loss.item():.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    no_compile = "--no-compile" in _smoke_args

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.set_float32_matmul_precision("high")

    try:
        test_eager_training_scale()
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
