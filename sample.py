"""
Sample from a saved checkpoint.
Usage: uv run sample.py [--prompt "your prompt"] [--tokens 200] [--temp 0.8] [--top-k 50]
"""

import argparse
import os

import torch
import torch.nn.functional as F

from prepare import MAX_SEQ_LEN, Tokenizer
from train import GPT, GPTConfig

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="checkpoint.pt")
parser.add_argument("--prompt", default="")
parser.add_argument("--tokens", type=int, default=200)
parser.add_argument("--temp", type=float, default=0.8)
parser.add_argument("--top-k", type=int, default=50)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

if not os.path.exists(args.checkpoint):
    raise FileNotFoundError(
        f"No checkpoint found at '{args.checkpoint}'. "
        "Make sure train.py has been run at least once."
    )

ckpt = torch.load(args.checkpoint, map_location=device)
config = GPTConfig(**ckpt["config"])
model = GPT(config).to(device)
# torch.compile prefixes all keys with "_orig_mod." — strip it if present
state_dict = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
model.load_state_dict(state_dict)
model = model.to(torch.bfloat16)
model.eval()
print(f"Loaded checkpoint from '{args.checkpoint}' ({config.n_layer}L, {config.n_embd}d)")

# ---------------------------------------------------------------------------
# Load tokenizer
# ---------------------------------------------------------------------------

tokenizer = Tokenizer.from_directory()

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(prompt: str, max_new_tokens: int, temperature: float, top_k: int) -> str:
    ids = tokenizer.encode(prompt) if prompt else [tokenizer.bos_token_id]
    ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

    for _ in range(max_new_tokens):
        ids_cond = ids[:, -MAX_SEQ_LEN:]
        logits = model(ids_cond)            # (1, T, vocab_size)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)

    return tokenizer.decode(ids[0].tolist())


print(f"\n--- prompt: {args.prompt!r} ---\n")
print(generate(args.prompt, max_new_tokens=args.tokens, temperature=args.temp, top_k=args.top_k))
