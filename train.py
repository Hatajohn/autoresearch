"""
Autoresearch pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
# Cache compiled Triton/CUDA kernels to a persistent directory so the cache
# survives reboots.  /tmp (the default) is wiped on every WSL/machine restart,
# forcing a full recompile each time.  Persistent cache cuts pre-warm from
# ~3000s cold (Ada Lovelace) / ~300s cold (H100) to ~10s warm on any GPU.
# Note: smoke tests compile a tiny model (n_embd=192, PROGRESSIVE window) and
# share no compiled kernel variants with the production config — running smoke
# tests before a training run does NOT pre-warm the production cache.
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",
                      os.path.expanduser("~/.cache/torchinductor"))

import gc
import math
import signal
import sys
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from kernels import get_kernel
    cap = torch.cuda.get_device_capability()
    # varunneal's FA3 is Hopper only, use kernels-community on non-Hopper GPUs
    repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
    fa3 = get_kernel(repo).flash_attn_interface
    print(f"Flash Attention 3 loaded (repo={repo}, cap={cap})", flush=True)
except Exception as e:
    print(f"Flash Attention 3 unavailable ({e}), falling back to torch SDPA.", flush=True)
    fa3 = None

from prepare import MAX_SEQ_LEN, Tokenizer, make_dataloader, evaluate_bpb

# All run constants are overridable via environment variables so the orchestration
# script can sweep experiments without touching this file (and without invalidating
# the torch.compile cache — graph structure is unchanged across hyperparameter sweeps).
TIME_BUDGET = int(os.environ.get("TRAIN_TIME_BUDGET", "360"))
RESUME_CHECKPOINT = os.environ.get("RESUME_CHECKPOINT", "")
VAL_INTERVAL = int(os.environ.get("VAL_INTERVAL", "0"))  # 0 = disabled


# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"
    ve_gate_channels: int = 0   # 0 = auto: max(32, n_embd // 16)
    pc_head_dim: int = 64       # bottleneck dim for PC prediction heads
    log_min_window: int = 0     # 0 = auto (seq // n_layer, tile-rounded); >0 overrides for LOG pattern


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    assert x.shape[3] % 2 == 0, f"head_dim must be even, got {x.shape[3]}"
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


def _make_causal_window_mask(T, window, device, dtype):
    """Additive causal mask with optional sliding window (0 = attend, -inf = mask)."""
    i = torch.arange(T, device=device).unsqueeze(1)   # (T, 1)
    j = torch.arange(T, device=device).unsqueeze(0)   # (1, T)
    causal = j > i                                    # upper triangle: future tokens
    if window > 0 and window < T:
        causal = causal | ((i - j) > window)          # also mask tokens beyond window
    return torch.zeros(T, T, device=device, dtype=dtype).masked_fill(causal, float('-inf'))


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        vgc = config.ve_gate_channels or max(32, config.n_embd // 16)
        self.ve_gate_channels = vgc
        self.ve_gate = nn.Linear(vgc, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
        # Learnable attention temperature (precision scalar): scales softmax sharpness.
        # β = attn_temp² + ε corresponds to inverse temperature in PC precision.
        # Initialized to 1 so scale ≈ 1/sqrt(head_dim) * 1 (neutral at step 0).
        self.attn_temp = nn.Parameter(torch.ones(1))

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        # Apply learnable temperature: higher β → sharper attention (higher precision).
        # Compute scale in float32 then cast to q's dtype: .square() on bf16 upcasts to
        # float32 under autocast, so the .to() must happen AFTER the arithmetic, not before.
        q = q * (self.attn_temp.square() + 0.01).to(dtype=q.dtype)

        if fa3 is not None:
            y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            q_ = q.transpose(1, 2)
            k_ = k.transpose(1, 2)
            v_ = v.transpose(1, 2)
            if self.n_kv_head != self.n_head:
                groups = self.n_head // self.n_kv_head
                k_ = k_.repeat_interleave(groups, dim=1)
                v_ = v_.repeat_interleave(groups, dim=1)
            window = window_size[0]
            if window < 0 or window >= T:
                y = F.scaled_dot_product_attention(q_, k_, v_, is_causal=True)
            else:
                mask = _make_causal_window_mask(T, window, q_.device, q_.dtype)
                y = F.scaled_dot_product_attention(q_, k_, v_, attn_mask=mask)
            y = y.transpose(1, 2)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class StochasticLayer(nn.Module):
    """Reparameterized sampling layer: makes layer uncertainty explicit.

    Each designated block outputs a posterior distribution q(z|x) = N(mu, sigma²)
    rather than a deterministic point. During training, z is sampled via the
    reparameterization trick; at eval, z = mu (no noise).

    KL divergence against N(0,1) is returned as a separate loss term, encouraging
    the model to maintain calibrated uncertainty estimates.
    """
    def __init__(self, n_embd):
        super().__init__()
        self.mu_proj       = nn.Linear(n_embd, n_embd, bias=False)
        self.log_sigma_proj = nn.Linear(n_embd, n_embd, bias=True)
        # Tensor flag: 1.0 in train, 0.0 in eval — avoids Python bool branch in forward.
        self.register_buffer("noise_scale", torch.ones(1))

    def train(self, mode=True):
        super().train(mode)
        # Keep noise_scale in sync with training mode so forward never branches on
        # a Python bool — torch.compile sees one static graph for both modes.
        self.noise_scale.fill_(1.0 if mode else 0.0)
        return self

    def forward(self, x):
        mu        = self.mu_proj(x)
        log_sigma = self.log_sigma_proj(x).clamp(-6.0, 2.0)
        sigma     = log_sigma.exp()
        # noise_scale=1 during training, 0 during eval — tensor branch, not Python bool,
        # so torch.compile emits one graph valid for both modes (no eval recompile).
        # Cast to mu's dtype so float32 buffer doesn't upcast bfloat16 activations under autocast.
        z = mu + self.noise_scale.to(dtype=mu.dtype) * sigma * torch.randn_like(mu)
        # KL(q || N(0,1)) = 0.5 * (sigma² + mu² - log(sigma²) - 1); float32 avoids cancellation
        kl = 0.5 * (sigma.float().pow(2) + mu.float().pow(2)
                     - 2.0 * log_sigma.float() - 1.0).mean()
        return z, kl


class TemporalSummarizer(nn.Module):
    """Soft temporal compression: mixes neighbouring token representations via a
    depthwise (channel-wise) causal convolution, then projects back.

    Keeps sequence length T constant so torch.compile fullgraph=True is preserved.
    The effect is that higher layers 'see' a blended version of local neighbourhoods,
    approximating the coarser-timescale summaries of a DTM without changing shapes.
    """
    def __init__(self, n_embd, stride=4):
        super().__init__()
        self.stride = stride
        # Depthwise conv: kernel_size=stride, causal padding on left only
        self.conv = nn.Conv1d(n_embd, n_embd, kernel_size=stride,
                              padding=stride - 1, groups=n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        # x: (B, T, C)  →  transpose for Conv1d  →  (B, C, T)
        B, T, C = x.size()
        h = x.transpose(1, 2)              # (B, C, T)
        h = self.conv(h)[..., :T]          # causal: keep first T outputs, drop padding
        h = h.transpose(1, 2)              # (B, T, C)
        h = self.proj(h)
        return x + h                       # residual: preserves original representation


class _NoVE(nn.Module):
    """Sentinel occupying non-VE slots in value_embeds ModuleList.
    Returns None so CausalSelfAttention skips the value residual path."""
    def forward(self, idx):
        return None


class _NoStoch(nn.Module):
    """Sentinel occupying non-stochastic slots in stochastic_layers ModuleList.
    Returns (x, zero) so the forward loop needs no conditional."""
    def __init__(self):
        super().__init__()
        self.register_buffer("_zero", torch.zeros(()))

    def forward(self, x):
        return x, self._zero


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        out = x + self.attn(norm(x), ve, cos_sin, window_size)
        out = out + self.mlp(norm(out))
        return out


AUX_HORIZONS = [(2, 0.15), (4, 0.05)]


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        # Soft temporal compression: inserted at the two tier-transition layers
        # (n//3 and 2*n//3) to blend neighbouring token representations before
        # higher-level (wider-window) blocks process them.  T stays constant.
        # nn.ModuleList with nn.Identity() at non-summarizer slots → forward loop
        # uses plain integer indexing with no conditionals.
        _summarizer_layers = {max(1, config.n_layer // 3), 2 * max(1, config.n_layer // 3)}
        self.summarizers = nn.ModuleList([
            TemporalSummarizer(config.n_embd) if i in _summarizer_layers else nn.Identity()
            for i in range(config.n_layer)
        ])
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.aux_lm_heads = nn.ModuleList([
            nn.Linear(config.n_embd, config.vocab_size, bias=False)
            for _ in AUX_HORIZONS
        ])
        # Learnable output scale for lm_head (log-space, always-positive via .exp()).
        # Initialised to 0.0 (scale=1) — neutral at init; wte_std controls the logit
        # magnitude directly.  Provides a per-run adaptable gain on the unembedding
        # without requiring a separate weight matrix.
        self.lm_head_log_scale = nn.Parameter(torch.zeros(()))
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        # Value embeddings — _NoVE sentinel at non-VE layers returns None, preserving the
        # existing `if ve is not None` guard in CausalSelfAttention without any dict lookup.
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleList([
            nn.Embedding(config.vocab_size, kv_dim) if has_ve(i, config.n_layer) else _NoVE()
            for i in range(config.n_layer)
        ])
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        # Focal gamma for output-difficulty weighting of pc_loss.
        # Stored as buffer (not literal) so sweeping PC_FOCAL_GAMMA never triggers recompile.
        self.register_buffer("pc_focal_gamma", torch.tensor(PC_FOCAL_GAMMA, dtype=torch.bfloat16),
                             persistent=False)
        # KL weight for stochastic layers.  Buffer → sweepable without recompile.
        self.register_buffer("kl_weight", torch.tensor(KL_WEIGHT, dtype=torch.bfloat16),
                             persistent=False)
        # Fixed normalisation for raw prediction errors (Bogacz 2017, Eq 10-11).
        # Dividing by sqrt(n_embd) makes errors dimensionless regardless of embedding scale.
        # The learned pc_lambdas² then represent precision = 1/Σ over these scaled errors,
        # rather than fighting the L2 norm of representations (previous pc_denom).
        self.register_buffer("pc_scale",
                             torch.tensor(config.n_embd ** 0.5, dtype=torch.bfloat16),
                             persistent=False)
        # Stochastic layers at every 3rd block — _NoStoch sentinel returns (x, zero_kl)
        # so the forward loop always unpacks a 2-tuple with no conditional branch.
        _stochastic_indices = set(range(2, config.n_layer, 3))  # 2, 5, 8, 11, ...
        self.stochastic_layers = nn.ModuleList([
            StochasticLayer(config.n_embd) if i in _stochastic_indices else _NoStoch()
            for i in range(config.n_layer)
        ])
        # Per-layer PC weights: simple positive scalars learned via softplus.
        # Using softplus(pc_log_lambdas) avoids the ELBO instability where a Gaussian
        # precision λ² → ∞ is rewarded whenever MSE < 1/λ², driving the loss to -∞.
        self.pc_log_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Stacked PC head weights — shape (n_layer, ...) so Phase 2 is a single
        # batched matmul instead of a Python loop over blocks.
        # pc_fc_w:   (n_layer, pc_head_dim, n_embd)  — projects n_embd → pc_head_dim (stored out×in)
        # pc_proj_w: (n_layer, pc_head_dim, n_embd)  — projects pc_head_dim → n_embd via transpose contraction
        self.pc_fc_w   = nn.Parameter(torch.zeros(config.n_layer, config.pc_head_dim, config.n_embd))
        self.pc_proj_w = nn.Parameter(torch.zeros(config.n_layer, config.pc_head_dim, config.n_embd))
        self.pc_gate_w = nn.Parameter(torch.zeros(config.n_layer, config.n_embd))  # (n_layer, C)
        self.pc_gate_b = nn.Parameter(torch.ones(config.n_layer))                  # (n_layer,) scalar bias per layer

    @torch.no_grad()
    def init_weights(self):
        # Embedding and unembedding
        # lm_head.weight is tied to wte.weight in __main__ — init wte only.
        # Target logit std ≈ 1.75 at init → E[CE] ≈ ln(vocab) + σ²/2 ≈ 9.01 + 1.5 ≈ 10.5 nats.
        # wte_std = σ_target / sqrt(n_embd): 1.75 / sqrt(512) ≈ 0.077.
        # (Previous 5.0/sqrt gave logit_std≈5, E[CE]≈21.5 nats — run 19 starting loss.)
        wte_std = 1.75 / (self.config.n_embd ** 0.5)
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=wte_std)
        for head in self.aux_lm_heads:
            torch.nn.init.normal_(head.weight, mean=0.0, std=0.001)
        # Transformer blocks — single pass covers attention, MLP, PC heads, and gates
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
            torch.nn.init.ones_(block.attn.attn_temp)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        # Stacked PC head weights — vectorised init outside block loop
        torch.nn.init.zeros_(self.pc_fc_w)         # zero-init matches pc_proj_w=0; residual pred grows smoothly
        torch.nn.init.zeros_(self.pc_proj_w)      # zero-init → pred = h_stack (identity residual) at step 0
        torch.nn.init.zeros_(self.pc_gate_w)
        self.pc_gate_b.fill_(1.0)                 # sigmoid(1) ≈ 0.73; most tokens active at init
        # Per-layer scalars
        # softplus(log(e-1)) = 1.0 exactly — neutral residual scale at init
        self.resid_lambdas.fill_(math.log(math.e - 1))
        self.pc_log_lambdas.fill_(0.0)   # softplus(0) ≈ 0.693 initial per-layer weight
        # Value embeddings (ModuleList contains nn.Embedding and _NoVE sentinels)
        for ve in self.value_embeds:
            if isinstance(ve, nn.Embedding):
                torch.nn.init.uniform_(ve.weight, -s, s)
        # StochasticLayer: mu_proj = identity (no distortion), log_sigma_proj = -3 (tiny sigma)
        for sl in self.stochastic_layers:
            if isinstance(sl, StochasticLayer):
                torch.nn.init.eye_(sl.mu_proj.weight)
                torch.nn.init.zeros_(sl.log_sigma_proj.weight)
                torch.nn.init.constant_(sl.log_sigma_proj.bias, -3.0)
            else:
                sl._zero.zero_()  # meta-device to_empty() does not guarantee zero init
        # TemporalSummarizer: zero-init proj so residual starts as identity
        for summ in self.summarizers:
            if isinstance(summ, TemporalSummarizer):
                torch.nn.init.zeros_(summ.proj.weight)
                torch.nn.init.ones_(summ.conv.weight)
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast embeddings to bf16
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds:
            if isinstance(ve, nn.Embedding):
                ve.to(dtype=torch.bfloat16)
        # Scalar buffers are zeroed by meta-device → to_empty() construction — restore
        # to their intended initial values.  Normal (non-meta) instantiation doesn't need
        # this, but it is idempotent so safe to call unconditionally.
        self.pc_focal_gamma.fill_(PC_FOCAL_GAMMA)
        self.kl_weight.fill_(KL_WEIGHT)
        self.pc_scale.fill_(float(self.config.n_embd) ** 0.5)
        self.lm_head_log_scale.fill_(0.0)  # neutral (scale=1); wte_std drives the logit magnitude
        for sl in self.stochastic_layers:
            if isinstance(sl, StochasticLayer):
                sl.noise_scale.fill_(1.0)   # start in training mode (1.0 = add noise)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        n = config.n_layer
        long_window = config.sequence_len
        if pattern == "PROGRESSIVE":
            # DTM-inspired: window grows monotonically with layer depth — 2 levels only.
            # Lower half of layers: short (half context); upper half: full context.
            # 2 levels instead of 4 halves the number of unique FA3 kernels to compile,
            # cutting cold-cache compilation time by ~50% on the attention kernels alone.
            short_window = long_window // 2
            window_sizes = []
            for i in range(n):
                w = short_window if i < n // 2 else long_window
                window_sizes.append((w, 0))
            # Final layer always uses full context regardless of pattern —
            # it produces the global broadcast target for predictive coding.
            window_sizes[-1] = (long_window, 0)
            return window_sizes
        if pattern == "LOG":
            # Logarithmic (geometric) scaling: each layer sees ~1.3× more context than
            # the previous.  Minimum window = seq // n_layer (≥ 64); maximum = seq.
            # Windows are rounded to the nearest 128 tokens for FA3 tile alignment.
            # Produces n unique kernel variants (one per layer), vs PROGRESSIVE's 2.
            # Trade-off: richer timescale hierarchy at the cost of a longer cold-cache
            # compile (paid once; subsequent runs use the kernel cache).
            tile = 128
            if config.log_min_window > 0:
                # Explicit override: use exactly the requested value.
                # tile_eff is clamped to min_w so the snap-to-tile inside the loop
                # cannot round 64 up to 128.
                v = config.log_min_window
                assert v >= 16 and (v & (v - 1)) == 0, (
                    f"log_min_window must be a power of 2 and >= 16, got {v}"
                )
                min_w = v
                tile_eff = min(tile, min_w)
            else:
                min_w = max(64, long_window // n)
                min_w = ((min_w + tile - 1) // tile) * tile  # round up to nearest tile
                tile_eff = tile
            window_sizes = []
            for i in range(n - 1):
                t = i / (n - 1)                                    # 0.0 → (n-2)/(n-1)
                w = min_w * (long_window / min_w) ** t             # geometric interpolation
                w = max(tile_eff, round(w / tile_eff) * tile_eff)  # snap to tile boundary
                window_sizes.append((w, 0))
            window_sizes.append((long_window, 0))        # last layer always full context
            return window_sizes
        assert all(c in "SL" for c in pattern)
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(n):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always uses full context regardless of pattern —
        # it produces the global broadcast target for predictive coding.
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds if isinstance(ve, nn.Embedding))
        nparams_exclude = (self.transformer.wte.weight.numel() + self.lm_head.weight.numel() +
                          value_embeds_numel)
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        # lm_head.weight is tied to wte.weight after weight tying — count 0 to avoid
        # double-counting the embedding matrix in the reported total.
        lm_head_tied = self.lm_head.weight is self.transformer.wte.weight
        lm_head = 0 if lm_head_tied else sum(p.numel() for p in self.lm_head.parameters())
        aux_heads = sum(p.numel() for p in self.aux_lm_heads.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.pc_log_lambdas.numel() + self.lm_head_log_scale.numel()
        summarizers = sum(p.numel() for p in self.summarizers.parameters())
        stochastic = sum(p.numel() for p in self.stochastic_layers.parameters())
        pc_heads = sum(p.numel() for p in [self.pc_fc_w, self.pc_proj_w, self.pc_gate_w, self.pc_gate_b])
        total = wte + value_embeds + lm_head + aux_heads + transformer_matrices + scalars + summarizers + stochastic + pc_heads
        return {
            'wte': wte, 'value_embeds': value_embeds,
            'lm_head': lm_head, 'lm_head_tied': lm_head_tied, 'aux_heads': aux_heads,
            'transformer_matrices': transformer_matrices, 'scalars': scalars,
            'summarizers': summarizers, 'stochastic': stochastic, 'pc_heads': pc_heads, 'total': total,
        }

    @torch.no_grad()
    def get_pc_diagnostics(self):
        """Return live scalar stats for the PC system — pure parameter reads, no forward pass."""
        n = self.config.n_layer
        pc_weights  = F.softplus(self.pc_log_lambdas).tolist()          # per-layer PC loss weight
        resid       = F.softplus(self.resid_lambdas).tolist()           # per-layer residual scale (effective)
        gate_biases = self.pc_gate_b.tolist()                            # learned gate threshold
        attn_temps  = [(block.attn.attn_temp.square() + 0.01).item()
                       for block in self.transformer.h]                  # effective attn temperature
        # Sigma bias: exp(log_sigma_proj.bias) gives baseline sigma when weight contribution
        # is small.  Stays near exp(-3)≈0.05 if StochasticLayer is collapsed (deterministic).
        sigma_bias_exp = [sl.log_sigma_proj.bias.exp().mean().item()
                          for sl in self.stochastic_layers
                          if isinstance(sl, StochasticLayer)]
        return {
            "pc_weights":     pc_weights,
            "resid":          resid,
            "gate_biases":    gate_biases,
            "attn_temps":     attn_temps,
            "sigma_bias_exp": sigma_bias_exp,
            "n_layer":        n,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5,
                        stoch_lr_scale=0.1):
        model_dim = self.config.n_embd
        pc_head_params = [self.pc_fc_w, self.pc_proj_w, self.pc_gate_w, self.pc_gate_b]
        pc_head_param_ids = {id(p) for p in pc_head_params}
        # attn_temp is a per-layer scalar — route to resid_params (high scalar LR)
        # rather than matrix_params (Muon optimizer, wrong for scalars).
        attn_temp_params = [block.attn.attn_temp for block in self.transformer.h]
        attn_temp_param_ids = {id(p) for p in attn_temp_params}
        exclude_ids = pc_head_param_ids | attn_temp_param_ids
        matrix_params = [p for p in self.transformer.h.parameters() if id(p) not in exclude_ids]
        value_embeds_params = list(self.value_embeds.parameters())
        # lm_head.weight is tied to wte.weight — counted once via embedding_params
        embedding_params = list(self.transformer.wte.parameters())
        aux_head_params = list(self.aux_lm_heads.parameters())
        summarizer_params = list(self.summarizers.parameters())
        stochastic_params = list(self.stochastic_layers.parameters())
        resid_params = [self.resid_lambdas, self.pc_log_lambdas, self.lm_head_log_scale] + attn_temp_params
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(aux_head_params) + len(value_embeds_params) + len(resid_params) +
            len(pc_head_params) + len(summarizer_params) + len(stochastic_params))
        # Scale LR ∝ 1/√dmodel (tuned at 768 dim)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind='adamw', params=aux_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            # pc head params in AdamW: kept out of Muon to avoid polluting backbone gradient groups
            # pc heads at 0.1× matrix_lr: cold auxiliary heads diverge at full backbone LR
            dict(kind='adamw', params=pc_head_params, lr=matrix_lr * 0.1, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            # summarizer matrices use AdamW (small, non-square conv/proj weights)
            dict(kind='adamw', params=summarizer_params, lr=matrix_lr, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            # stochastic layer matrices at a fraction of matrix_lr: Adam sign-normalises
            # gradients so each element shifts by ±lr per step; at matrix_lr=0.04 this
            # moves mu by ~0.04*sqrt(n_embd)≈0.9 per element in one step, destabilising
            # the representations that PC loss measures.  stoch_lr_scale=0.1 keeps the
            # per-step shift to ~0.09 per element, preserving near-identity init behaviour.
            dict(kind='adamw', params=stochastic_params, lr=matrix_lr * stoch_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx)
        x = norm(x)
        kl_loss = torch.zeros((), dtype=torch.float32, device=x.device)  # float32: matches StochasticLayer KL dtype
        # Pre-compute per-layer scalars outside the loop so torch.compile sees a single
        # tensor slice op rather than n_layer separate scalar-index ops.
        resid_scales = F.softplus(self.resid_lambdas).unbind(0)

        # ── Phase 1: standard forward pass ───────────────────────────────────
        # Run all layers, collecting their outputs.  No PC corrections yet —
        # the broadcast target (final layer) must exist before any error can
        # be computed.
        layer_outs = []
        for i, block in enumerate(self.transformer.h):
            x = resid_scales[i] * x
            x = self.summarizers[i](x)                    # Identity at non-summarizer layers
            ve = self.value_embeds[i](idx)                # _NoVE returns None at non-VE layers
            x = block(x, ve, cos_sin, self.window_sizes[i])
            x, kl_contrib = self.stochastic_layers[i](x)  # _NoStoch returns (x, 0) at non-stoch layers
            kl_loss = kl_loss + kl_contrib
            layer_outs.append(x)

        # ── Phase 2: hierarchical predictive coding ─────────────────────────
        # Layer i predicts norm(layer_outs[i+1]) — its immediate neighbour,
        # not a single global broadcast.  Each target is detached so the upper
        # layer cannot receive gradient through this path; only the predictor
        # (lower layer) trains on the error.
        #
        # The routing gate weights which tokens receive a strong PC signal.
        # At sigmoid(bias=1)≈0.73 initialisation most tokens contribute; the gate
        # learns to focus or suppress.  Gradient flows back through the weighted-MSE
        # loss (stable softplus layer weights), not through an explicit residual
        # correction (which would require a second forward pass).
        L = self.config.n_layer - 1
        normed_outs   = [norm(layer_outs[i]) for i in range(self.config.n_layer)]
        final_normed  = normed_outs[-1]                                                          # reuse; avoids a redundant norm() call
        h_stack       = torch.stack(normed_outs[:L])                                             # (L, B, T, C) — predictors
        targets_stack = torch.stack([normed_outs[i + 1].detach() for i in range(L)])            # (L, B, T, C) — per-layer targets

        # Residual pred_head: pred = h + proj(tanh(fc(h)))
        fc_out  = torch.einsum('lbtc,ldc->lbtd', h_stack, self.pc_fc_w[:L])              # (L, B, T, pc_head_dim)
        pred    = h_stack + torch.einsum('lbtd,ldc->lbtc', torch.tanh(fc_out), self.pc_proj_w[:L])  # (L, B, T, C)

        raw_error = (targets_stack - pred) / self.pc_scale                                # (L, B, T, C)
        pc_w      = F.softplus(self.pc_log_lambdas[:L]).view(L, 1, 1)                    # (L, 1, 1)
        token_pc  = pc_w * raw_error.pow(2).mean(dim=-1)                                 # (L, B, T)

        # Gate: per-layer linear → sigmoid.  pc_gate_w is (n_layer, C), pc_gate_b is (n_layer,)
        gate_logit = torch.einsum('lbtc,lc->lbt', h_stack.detach(), self.pc_gate_w[:L])   # (L, B, T); detach prevents backbone from nulling the gate
        gate       = torch.sigmoid(gate_logit + self.pc_gate_b[:L].view(L, 1, 1))        # (L, B, T)

        pc_loss_map = (gate * token_pc).sum(dim=0)                                        # (B, T)

        x = final_normed

        softcap = 15
        logits = self.lm_head(x) * self.lm_head_log_scale.exp()
        # Keep logits in bfloat16 — F.cross_entropy promotes internally for log-sum-exp,
        # so the explicit float32 cast only wastes ~2GB of GPU memory per forward+backward.
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            # Compute per-token CE once; derive scalar loss and focal map from the same tensor.
            # One kernel instead of two (old code called cross_entropy with reduction='mean'
            # AND reduction='none' separately, generating two distinct compiled kernel variants).
            token_ce = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                ignore_index=-1, reduction='none',
            ).view(B, T)
            if reduction == 'mean':
                loss = token_ce.mean()
                # Multi-timescale auxiliary losses (Parr et al. Section 5.2): predict k tokens ahead.
                # Each horizon uses a dedicated unembedding head to avoid gradient interference.
                # Only added during training (reduction='mean'); evaluate_bpb uses reduction='none'.
                for head, (horizon_k, horizon_w) in zip(self.aux_lm_heads, AUX_HORIZONS):
                    aux_logits = head(x[:, :-horizon_k])
                    aux = F.cross_entropy(
                        aux_logits.contiguous().view(-1, aux_logits.size(-1)),
                        targets[:, horizon_k:].contiguous().view(-1),
                        ignore_index=-1, reduction='mean',
                    )
                    loss = loss + horizon_w * aux
            else:
                loss = token_ce  # (B, T) per-token losses — used by evaluate_bpb
            # Output-difficulty focal weighting: tokens where the model is most wrong
            # get the largest PC correction signal.  Fully detached — only scales magnitude.
            tc_det = token_ce.detach()
            difficulty = (tc_det / (tc_det.mean() + 1e-6)).pow(self.pc_focal_gamma)
            pc_loss = (pc_loss_map * difficulty).mean() / max(1, len(self.transformer.h) - 1)
            # Fold kl_loss into pc_loss before returning so callers (including
            # evaluate_bpb in prepare.py) always receive a 2-tuple (loss, aux_loss).
            aux_loss = pc_loss + self.kl_weight * kl_loss
            return loss, aux_loss
        return logits

# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    # NorMuon variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128          # target head dimension for attention
WINDOW_PATTERN = "LOG"          # Geometric: each layer sees ~1.3x more context than the previous

# Optimization
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
EMBEDDING_LR = 0.1      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Predictive coding — readable from env so orchestrator can sweep without recompile
PC_WEIGHT        = float(os.environ.get("PC_WEIGHT",        "0.1"))
PC_WEIGHT_WARMUP = int(os.environ.get("PC_WEIGHT_WARMUP",  "50"))   # steps to ramp PC_WEIGHT from 0→full (0 = off)
PC_FOCAL_GAMMA   = float(os.environ.get("PC_FOCAL_GAMMA",  "1.0"))  # 0.0 = uniform (no focal)
KL_WEIGHT        = float(os.environ.get("KL_WEIGHT",       "0.01")) # KL div from stochastic layers
PC_DIAG_INTERVAL = int(os.environ.get("PC_DIAG_INTERVAL",  "50"))   # steps between PC diagnostic lines (0 = off)

# Model size
DEPTH = 8               # number of transformer layers
DEVICE_BATCH_SIZE = 32   # per-device batch size (RTX 4080 16GB; H100 default was 128)
PC_HEAD_DIM = int(os.environ.get("PC_HEAD_DIM", str(max(64, DEPTH * ASPECT_RATIO // 8))))
LOG_MIN_WINDOW = int(os.environ.get("LOG_MIN_WINDOW", "0"))  # 0 = auto; set e.g. 64 to override

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Graceful shutdown: release GPU memory and flush logs when the process is
    # terminated externally (SIGTERM from orchestrator or Ctrl-C / SIGINT).
    def _handle_signal(signum, frame):
        sig_name = signal.Signals(signum).name
        print(f"\n[train] Received {sig_name} — flushing GPU and exiting cleanly.", flush=True)
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT,  _handle_signal)

    t_start = time.time()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.cache_size_limit = 64
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    # MFU is reported relative to H100 SXM5 BF16 peak (989.5 TFLOPS) as a fixed
    # reference point so numbers are comparable across runs regardless of hardware.
    # On Ada Lovelace (RTX 4080/4090, ~165 TFLOPS BF16) true utilisation is ~6x higher.
    REFERENCE_BF16_PEAK_FLOPS = 989.5e12

    print("Loading tokenizer...", flush=True)
    _t0 = time.time()
    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()
    t_tokenizer = time.time() - _t0
    print(f"Vocab size: {vocab_size:,} ({t_tokenizer:.1f}s)", flush=True)

    def build_model_config(depth):
        base_dim = depth * ASPECT_RATIO
        model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
        num_heads = model_dim // HEAD_DIM
        return GPTConfig(
            sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
            n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
            window_pattern=WINDOW_PATTERN,
            pc_head_dim=PC_HEAD_DIM,
            log_min_window=LOG_MIN_WINDOW,
        )

    config = build_model_config(DEPTH)
    print(f"Model config: {asdict(config)}")

    print("Initializing model...", flush=True)
    _t0 = time.time()
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    t_model_init = time.time() - _t0
    print(f"  model init done ({t_model_init:.1f}s)", flush=True)

    # Weight tying: lm_head shares weights with the input embedding.
    # Must happen before num_scaling_params() (to report the correct unique count)
    # and before setup_optimizer() (so the tied tensor is not in two param groups).
    model.lm_head.weight = model.transformer.wte.weight

    param_counts = model.num_scaling_params()
    print("Parameter counts:")
    for key, value in param_counts.items():
        print(f"  {key:24s}: {value:,}")
    num_params = param_counts['total']
    num_flops_per_token = model.estimate_flops()
    print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

    tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
    assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
    grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

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
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt.get("step", 0)
        total_training_time = ckpt.get("total_training_time", 0.0)
        print(f"Resumed from {RESUME_CHECKPOINT} at step {step}", flush=True)

    print("Compiling model (first run: slow, cached thereafter)...", flush=True)
    _t0 = time.time()
    model = torch.compile(model, dynamic=False, fullgraph=True)
    t_compile = time.time() - _t0
    print(f"  torch.compile done ({t_compile:.1f}s)", flush=True)

    # Pre-warm: run one dummy forward+backward to trigger all Triton kernel compilations
    # before the training loop starts.  This makes step 0 fast and keeps the "Starting
    # training loop" message honest — all compilation noise is absorbed here.
    print("Pre-warming kernels (dummy fwd+bwd — ~10s warm / up to ~3000s cold on Ada Lovelace)...", flush=True)
    _t_prewarm = time.time()
    _dummy_x = torch.zeros(DEVICE_BATCH_SIZE, MAX_SEQ_LEN, dtype=torch.long, device=device)
    _dummy_y = torch.zeros(DEVICE_BATCH_SIZE, MAX_SEQ_LEN, dtype=torch.long, device=device)
    with autocast_ctx:
        _d_loss, _d_aux = model(_dummy_x, _dummy_y)
        (_d_loss + PC_WEIGHT * _d_aux).backward()
    model.zero_grad(set_to_none=True)
    del _dummy_x, _dummy_y, _d_loss, _d_aux
    torch.cuda.synchronize()
    t_prewarm = time.time() - _t_prewarm
    print(f"  kernel pre-warm done ({t_prewarm:.0f}s)", flush=True)

    # Flush memory fragmentation left by compile workers before training starts.
    torch.cuda.empty_cache()
    print(f"GPU memory after compile: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated, "
          f"{torch.cuda.memory_reserved()/1e9:.2f}GB reserved", flush=True)

    print("Prefetching first batch...", flush=True)
    train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
    x, y, epoch = next(train_loader)  # prefetch first batch

    print(f"Time budget: {TIME_BUDGET}s", flush=True)
    print(f"Gradient accumulation steps: {grad_accum_steps}", flush=True)
    print("Starting training loop...", flush=True)

    # Schedules (all based on progress = training_time / TIME_BUDGET)

    def get_lr_multiplier(progress):
        if progress < WARMUP_RATIO:
            return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
        elif progress < 1.0 - WARMDOWN_RATIO:
            return 1.0
        else:
            cooldown = (1.0 - progress) / WARMDOWN_RATIO
            return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

    def get_muon_momentum(step):
        frac = min(step / 300, 1)
        return (1 - frac) * 0.85 + frac * 0.95

    def get_weight_decay(progress):
        return WEIGHT_DECAY * (1 - progress)

    KL_WARMUP_STEPS = 200  # ramp kl_weight from 0 → KL_WEIGHT over this many steps
    def get_kl_weight(step):
        return KL_WEIGHT * min(step / KL_WARMUP_STEPS, 1.0)

    def get_pc_weight(step):
        """Linear warmup of PC_WEIGHT from 0 → PC_WEIGHT over PC_WEIGHT_WARMUP steps.
        Lets the backbone stabilize before the PC signal is introduced.
        PC_WEIGHT_WARMUP=0 disables warmup (full weight from step 0)."""
        if PC_WEIGHT_WARMUP <= 0:
            return PC_WEIGHT
        return PC_WEIGHT * min(step / PC_WEIGHT_WARMUP, 1.0)

    # ---------------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------------

    t_start_training = time.time()
    smooth_train_loss = 0
    # step and total_training_time already initialised (and possibly restored from checkpoint)

    while True:
        torch.cuda.synchronize()
        t0 = time.time()
        model.kl_weight.fill_(get_kl_weight(step))
        train_loss_accum = torch.zeros((), device=device)
        train_pc_loss_accum = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            with autocast_ctx:
                main_loss, aux_loss = model(x, y)
                # aux_loss already contains pc_loss + kl_weight*kl_loss (folded in forward)
                loss = main_loss + get_pc_weight(step) * aux_loss
            train_loss_accum += main_loss.detach()
            train_pc_loss_accum += aux_loss.detach()
            loss = loss / grad_accum_steps
            loss.backward()
            x, y, epoch = next(train_loader)
        train_loss = train_loss_accum / grad_accum_steps
        train_pc_loss = train_pc_loss_accum / grad_accum_steps

        # Progress and schedules
        progress = min(total_training_time / TIME_BUDGET, 1.0)
        lrm = get_lr_multiplier(progress)
        muon_momentum = get_muon_momentum(step)
        muon_weight_decay = get_weight_decay(progress)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group['kind'] == 'muon':
                group["momentum"] = muon_momentum
                group["weight_decay"] = muon_weight_decay
        # Clip gradients before the optimizer step.  Muon's NorMuon normalisation
        # already bounds backbone matrix steps; this catches runaway gradients on the
        # AdamW-trained params (pred_head, routing_gate, summarizers, stochastic layers).
        # clip_grad_norm_ returns the pre-clip total norm.
        all_params = [p for g in optimizer.param_groups for p in g['params']]
        grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        optimizer.step()
        model.zero_grad(set_to_none=True)

        train_loss_f = train_loss.item()
        train_pc_loss_f = train_pc_loss.item()

        # Fast fail: abort if loss is exploding or NaN
        if math.isnan(train_loss_f) or train_loss_f > 100:
            print(f"FAIL at step {step}, loss={train_loss_f:.4f}, lrm={lrm:.4f}", flush=True)
            torch.save({"model": model.state_dict(), "config": asdict(config), "step": step},
                       "checkpoint_failed.pt")
            sys.exit(1)

        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0

        if step > 10:
            total_training_time += dt

        # Logging
        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
        pct_done = 100 * progress
        tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
        mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / REFERENCE_BF16_PEAK_FLOPS
        remaining = max(0, TIME_BUDGET - total_training_time)

        kl_w = model.kl_weight.item()
        print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | pc_loss: {train_pc_loss_f:.6f} | kl_w: {kl_w:.4f} | grad_norm: {grad_norm:.3f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

        if PC_DIAG_INTERVAL > 0 and (step + 1) % PC_DIAG_INTERVAL == 0:
            d = model.get_pc_diagnostics()
            n = d["n_layer"]
            # Phase 2 loops over layers 0..n-2; last layer is broadcast target only.
            # Mark it with "-" so it's visually clear it has no PC contribution.
            def _fmt(vals, last_marker=True):
                parts = [f"{v:.2f}" for v in vals[:-1]]
                parts.append("  -- " if last_marker else f"{vals[-1]:.2f}")
                return "[" + "  ".join(parts) + "]"
            sigma_str = "  ".join(f"{v:.3f}" for v in d['sigma_bias_exp'])
            print(f"\n  pc_diag step {step + 1:05d}"
                  f"\n    pc_weights : {_fmt(d['pc_weights'])}"
                  f"\n    gate_biases: {_fmt(d['gate_biases'])}"
                  f"\n    resid_λ    : {_fmt(d['resid'], last_marker=False)}"
                  f"\n    attn_temps : {_fmt(d['attn_temps'], last_marker=False)}"
                  f"\n    sigma_bias : [{sigma_str}]  (exp of log_sigma bias; ~0.05=collapsed, ~1.0=active)",
                  flush=True)

        if VAL_INTERVAL > 0 and step > 0 and step % VAL_INTERVAL == 0:
            model.eval()
            with autocast_ctx, torch.no_grad():
                mid_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)
            model.train()
            print(f"\n  val_bpb (step {step}): {mid_bpb:.6f}", flush=True)

        # GC management (Python's GC causes ~500ms stalls)
        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif (step + 1) % 5000 == 0:
            gc.collect()

        step += 1

        # Time's up — but only stop after warmup steps so we don't count compilation
        if step > 10 and total_training_time >= TIME_BUDGET:
            break

    print()  # newline after \r training log

    total_tokens = step * TOTAL_BATCH_SIZE

    # Final eval — torch.no_grad() prevents building a computation graph during
    # validation, saving ~30% of forward-pass time and freeing activation memory.
    model.eval()
    with autocast_ctx, torch.no_grad():
        val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

    # Final summary
    t_end = time.time()
    startup_time = t_start_training - t_start
    steady_state_mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10) / total_training_time / REFERENCE_BF16_PEAK_FLOPS if total_training_time > 0 else 0
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    ckpt_path = "checkpoint.pt"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(config),
        "step": step,
        "total_training_time": total_training_time,
    }, ckpt_path)

    print("---")
    print(f"val_bpb:          {val_bpb:.6f}")
    print(f"startup_seconds:  {startup_time:.1f} (tokenizer={t_tokenizer:.1f}s, init={t_model_init:.1f}s, compile={t_compile:.1f}s, prewarm={t_prewarm:.0f}s)")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"mfu_percent:      {steady_state_mfu:.2f}")
    print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.1f}")
    print(f"depth:            {DEPTH}")
