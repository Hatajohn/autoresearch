"""
GPU cooldown between experiments.
Clears CUDA cache, forces GC, prints VRAM stats, then sleeps 60s.
Usage: uv run cooldown.py
"""
import gc
import time
import torch

print("--- GPU Cooldown ---")
print(f"Before: VRAM allocated = {torch.cuda.memory_allocated()/1024**2:.0f} MB  "
      f"reserved = {torch.cuda.memory_reserved()/1024**2:.0f} MB")

gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

print(f"After:  VRAM allocated = {torch.cuda.memory_allocated()/1024**2:.0f} MB  "
      f"reserved = {torch.cuda.memory_reserved()/1024**2:.0f} MB")

print("Cooling down for 60s...", flush=True)
for i in range(60, 0, -10):
    print(f"  {i}s remaining...", flush=True)
    time.sleep(10)

print("Cooldown complete. Ready for next run.")
