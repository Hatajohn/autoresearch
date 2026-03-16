"""
Autoresearch pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"  # flush stdout immediately even when redirected to file
# Cache compiled Triton/CUDA kernels to disk so subsequent runs skip recompilation entirely.
# This eliminates the recurring 100-300s stalls that burn training budget.
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")

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

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

# All run constants are overridable via environment variables so the orchestration
# script can sweep experiments without touching this file (and without invalidating
# the torch.compile cache — graph structure is identical across all PC_ALPHA values).
TIME_BUDGET = int(os.environ.get("TRAIN_TIME_BUDGET", "360"))


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


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


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
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
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
            y = F.scaled_dot_product_attention(q_, k_, v_, is_causal=True)
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


class PredHead(nn.Module):
    """Two-layer MLP predictor: out → hidden → prediction of x."""
    def __init__(self, n_embd):
        super().__init__()
        self.fc   = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        return x + self.proj(F.tanh(self.fc(x)))


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
        self.log_sigma_proj = nn.Linear(n_embd, n_embd, bias=False)
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
        # KL(q || N(0,1)) = 0.5 * (sigma² + mu² - log(sigma²) - 1)
        kl = 0.5 * (sigma.pow(2) + mu.pow(2) - 2.0 * log_sigma - 1.0).mean()
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


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
        self.pred_head = PredHead(config.n_embd)
        # Per-token precision gate: learns which token positions should receive
        # top-down PC correction.  Initialized to zero → sigmoid(0)=0.5 (neutral).
        self.routing_gate = nn.Linear(config.n_embd, 1, bias=False)

    def forward(self, x, ve, cos_sin, window_size):
        out = x + self.attn(norm(x), ve, cos_sin, window_size)
        out = out + self.mlp(norm(out))
        return out


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
        third = max(1, config.n_layer // 3)
        self.summarizer_layers = {third, 2 * third}
        self.summarizers = nn.ModuleDict({
            str(i): TemporalSummarizer(config.n_embd)
            for i in self.summarizer_layers
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        # Value embeddings
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        # PC routing strength as a non-persistent buffer so torch.compile sees a
        # tensor placeholder rather than a Python literal — changing PC_ALPHA between
        # experiments never invalidates the compiled graph.
        self.register_buffer("pc_alpha", torch.tensor(PC_ALPHA, dtype=torch.bfloat16),
                             persistent=False)
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
        # Stochastic layers at every 3rd block to add explicit uncertainty
        stochastic_indices = set(range(2, config.n_layer, 3))  # 2, 5, 8, 11, ...
        self.stochastic_layers = nn.ModuleDict({
            str(i): StochasticLayer(config.n_embd)
            for i in stochastic_indices
        })
        # Integer-indexed sets for O(1) membership checks in forward — avoids
        # str(i) conversion on every iteration inside the compiled loop.
        self.stochastic_layer_set = stochastic_indices
        self.value_embed_set = {i for i in range(config.n_layer) if has_ve(i, config.n_layer)}
        # Per-layer learnable precision scalars: weight each layer's PC error contribution.
        # Initialized to 1 (uniform), learned to up/down-weight layers during training.
        self.pc_lambdas = nn.Parameter(torch.ones(config.n_layer))
        # Skip-layer PC schedule: shallow layers predict 1 step back, deeper layers predict
        # 2 or 3 steps back, creating genuinely long-range top-down signals.
        # Divides n_layer into thirds; skip distance increases by 1 each third.
        third = max(1, config.n_layer // 3)
        self.skip_schedule = [1 + (i // third) for i in range(config.n_layer)]
        self.history_depth = max(self.skip_schedule) + 1

    @torch.no_grad()
    def init_weights(self):
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
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
            torch.nn.init.uniform_(block.pred_head.fc.weight, -s, s)
            torch.nn.init.zeros_(block.pred_head.proj.weight)
            torch.nn.init.zeros_(block.routing_gate.weight)
        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.pc_lambdas.fill_(1.0)
        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        # StochasticLayer: mu_proj = identity (no distortion), log_sigma_proj = -3 (tiny sigma)
        for sl in self.stochastic_layers.values():
            torch.nn.init.eye_(sl.mu_proj.weight)
            torch.nn.init.constant_(sl.log_sigma_proj.weight, -3.0 / sl.log_sigma_proj.weight.size(0))
        # TemporalSummarizer: zero-init proj so residual starts as identity
        for summ in self.summarizers.values():
            torch.nn.init.zeros_(summ.proj.weight)
            torch.nn.init.ones_(summ.conv.weight)
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast embeddings to bf16
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)
        # Scalar buffers are zeroed by meta-device → to_empty() construction — restore
        # to their intended initial values.  Normal (non-meta) instantiation doesn't need
        # this, but it is idempotent so safe to call unconditionally.
        self.pc_alpha.fill_(PC_ALPHA)
        self.pc_focal_gamma.fill_(PC_FOCAL_GAMMA)
        self.kl_weight.fill_(KL_WEIGHT)
        self.pc_scale.fill_(float(self.config.n_embd) ** 0.5)
        for sl in self.stochastic_layers.values():
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
            min_w = max(64, long_window // n)
            min_w = ((min_w + tile - 1) // tile) * tile  # round up to nearest tile
            window_sizes = []
            for i in range(n - 1):
                t = i / (n - 1)                          # 0.0 → (n-2)/(n-1)
                w = min_w * (long_window / min_w) ** t   # geometric interpolation
                w = max(tile, round(w / tile) * tile)    # snap to tile boundary
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
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel())
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
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.pc_lambdas.numel()
        summarizers = sum(p.numel() for p in self.summarizers.parameters())
        stochastic = sum(p.numel() for p in self.stochastic_layers.parameters())
        total = wte + value_embeds + lm_head + transformer_matrices + scalars + summarizers + stochastic
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'scalars': scalars,
            'summarizers': summarizers, 'stochastic': stochastic, 'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        pred_head_params = [p for block in self.transformer.h
                            for p in list(block.pred_head.parameters())
                                     + list(block.routing_gate.parameters())]
        pred_head_param_ids = {id(p) for p in pred_head_params}
        # attn_temp is a per-layer scalar — route to resid_params (high scalar LR)
        # rather than matrix_params (Muon optimizer, wrong for scalars).
        attn_temp_params = [block.attn.attn_temp for block in self.transformer.h]
        attn_temp_param_ids = {id(p) for p in attn_temp_params}
        exclude_ids = pred_head_param_ids | attn_temp_param_ids
        matrix_params = [p for p in self.transformer.h.parameters() if id(p) not in exclude_ids]
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        summarizer_params = list(self.summarizers.parameters())
        stochastic_params = list(self.stochastic_layers.parameters())
        resid_params = [self.resid_lambdas, self.pc_lambdas] + attn_temp_params
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) +
            len(pred_head_params) + len(summarizer_params) + len(stochastic_params))
        # Scale LR ∝ 1/√dmodel (tuned at 768 dim)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            # pred_head in AdamW: kept out of Muon to avoid polluting backbone gradient groups
            dict(kind='adamw', params=pred_head_params, lr=matrix_lr, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            # summarizer matrices use AdamW (small, non-square conv/proj weights)
            dict(kind='adamw', params=summarizer_params, lr=matrix_lr, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            # stochastic layer matrices (mu_proj, log_sigma_proj) use AdamW
            dict(kind='adamw', params=stochastic_params, lr=matrix_lr, betas=adam_betas, eps=1e-10, weight_decay=0.0),
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
        pc_loss_map = x.new_zeros(B, T)  # (B, T) per-token accumulator; reduced after logits
        kl_loss = x.new_zeros(())        # scalar KL accumulated across stochastic layers
        # History buffer for skip-layer PC: history[-1] is most recent, history[-k] is k steps back.
        # Pre-fill all slots with the initial embedding so every layer has a valid target
        # regardless of skip distance.  Static length = max_skip + 1 → compile-safe.
        history = [x.detach()] * self.history_depth
        # Pre-compute per-layer scalars outside the loop so torch.compile sees a single
        # tensor slice op rather than n_layer separate scalar-index ops, avoiding recompilation.
        resid_scales = self.resid_lambdas.unbind(0)
        lambda_sq = self.pc_lambdas.square()
        pc_weights          = lambda_sq.unbind(0)          # λᵢ² — precision, used in loss
        pc_log_weights      = lambda_sq.clamp(min=1e-8).log().unbind(0)  # log(λᵢ²) — ELBO term
        pc_precision_routing = lambda_sq.detach().unbind(0)               # λᵢ² detached — routing
        for i, block in enumerate(self.transformer.h):
            x = resid_scales[i] * x
            # Soft temporal compression at tier-transition layers (before the block)
            if i in self.summarizer_layers:
                x = self.summarizers[str(i)](x)
            ve = self.value_embeds[str(i)](idx) if i in self.value_embed_set else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
            # Skip-layer PC: shallow layers predict 1 step back (adjacent), deep layers
            # predict 2-3 steps back — increasingly long-range top-down signals.
            skip = self.skip_schedule[i]   # static Python int at compile time
            target_out = history[-skip]    # static negative index; no dynamic dispatch
            pred = block.pred_head(norm(x))
            # Bogacz (2017) Eq 10-11: errors are normalised by Σ (variance), not ||target||.
            # We use pc_scale = sqrt(n_embd) as a fixed dim-normaliser; λ² = 1/Σ is learned.
            raw_error = (target_out - pred) / self.pc_scale    # (B, T, C), dimensionless
            # Loss path: ELBO = λ² * error² - log(λ²)
            # Without -log(λ²), λ grows unbounded; with it, λ² → 1/E[error²] at equilibrium.
            pc_contrib = (pc_weights[i] * raw_error.pow(2).mean(dim=-1)
                          - pc_log_weights[i])                 # (B, T)
            pc_loss_map = pc_loss_map + pc_contrib
            # Routing path: precision-weighted correction — λ² · error (Bogacz Eq 53).
            # gate ∈ (0,1); sigmoid(0)=0.5 initially → neutral, learns where to correct.
            gate = torch.sigmoid(block.routing_gate(norm(x)))   # (B, T, 1)
            x = x + gate * self.pc_alpha * pc_precision_routing[i] * (target_out - pred.detach()) / self.pc_scale
            # Stochastic reparameterization at designated layers
            if i in self.stochastic_layer_set:
                x, kl_contrib = self.stochastic_layers[str(i)](x)
                kl_loss = kl_loss + kl_contrib
            history = history[1:] + [x.detach()]  # rotate: drop oldest, append current
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
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
                # Only added during training (reduction='mean'); evaluate_bpb uses reduction='none'
                # and the shifted-target windows produce different-sized tensors that can't be summed
                # with the per-token (B, T) map.
                for horizon_k, horizon_w in ((2, 0.15), (4, 0.05)):
                    aux = F.cross_entropy(
                        logits[:, :-horizon_k].contiguous().view(-1, logits.size(-1)),
                        targets[:, horizon_k:].contiguous().view(-1),
                        ignore_index=-1, reduction='mean',
                    )
                    loss = loss + horizon_w * aux
            else:
                loss = token_ce  # (B, T) per-token losses — used by evaluate_bpb
            # Output-difficulty focal weighting: tokens where the model is most wrong
            # get the largest PC correction signal.  Fully detached — only scales magnitude.
            difficulty = (token_ce.detach() / (token_ce.detach().mean() + 1e-6)).pow(self.pc_focal_gamma)
            pc_loss = (pc_loss_map * difficulty).mean() / len(self.transformer.h)
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
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Predictive coding — readable from env so orchestrator can sweep without recompile
PC_WEIGHT       = float(os.environ.get("PC_WEIGHT",       "0.1"))
PC_ALPHA        = float(os.environ.get("PC_ALPHA",        "0.1"))  # 0.0 = Option A (no routing)
PC_FOCAL_GAMMA  = float(os.environ.get("PC_FOCAL_GAMMA",  "1.0"))  # 0.0 = uniform (no focal)
KL_WEIGHT       = float(os.environ.get("KL_WEIGHT",       "0.01")) # KL div from stochastic layers

# Model size
DEPTH = 8               # number of transformer layers
DEVICE_BATCH_SIZE = 32   # per-device batch size (RTX 4080 16GB; H100 default was 128)

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
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    H100_BF16_PEAK_FLOPS = 989.5e12

    print("Loading tokenizer...", flush=True)
    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size:,}", flush=True)

    def build_model_config(depth):
        base_dim = depth * ASPECT_RATIO
        model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
        num_heads = model_dim // HEAD_DIM
        return GPTConfig(
            sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
            n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
            window_pattern=WINDOW_PATTERN,
        )

    config = build_model_config(DEPTH)
    print(f"Model config: {asdict(config)}")

    print("Initializing model...", flush=True)
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    print(f"  model init done", flush=True)

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

    print("Compiling model (first run: slow, cached thereafter)...", flush=True)
    model = torch.compile(model, dynamic=False, fullgraph=True)

    # Pre-warm: run one dummy forward+backward to trigger all Triton kernel compilations
    # before the training loop starts.  This makes step 0 fast and keeps the "Starting
    # training loop" message honest — all compilation noise is absorbed here.
    print("Pre-warming kernels (dummy fwd+bwd — may take several minutes on cold cache)...", flush=True)
    _t_prewarm = time.time()
    _dummy_x = torch.zeros(DEVICE_BATCH_SIZE, MAX_SEQ_LEN, dtype=torch.long, device=device)
    _dummy_y = torch.zeros(DEVICE_BATCH_SIZE, MAX_SEQ_LEN, dtype=torch.long, device=device)
    with autocast_ctx:
        _d_loss, _d_aux = model(_dummy_x, _dummy_y)
        (_d_loss + PC_WEIGHT * _d_aux).backward()
    model.zero_grad(set_to_none=True)
    del _dummy_x, _dummy_y, _d_loss, _d_aux
    torch.cuda.synchronize()
    print(f"  kernel pre-warm done in {time.time() - _t_prewarm:.0f}s", flush=True)

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

    # ---------------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------------

    t_start_training = time.time()
    smooth_train_loss = 0
    total_training_time = 0
    step = 0

    while True:
        torch.cuda.synchronize()
        t0 = time.time()
        for micro_step in range(grad_accum_steps):
            with autocast_ctx:
                main_loss, aux_loss = model(x, y)
                # aux_loss already contains pc_loss + kl_weight*kl_loss (folded in forward)
                loss = main_loss + PC_WEIGHT * aux_loss
            train_loss = main_loss.detach()
            train_pc_loss = aux_loss.detach()
            loss = loss / grad_accum_steps
            loss.backward()
            x, y, epoch = next(train_loader)

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
        optimizer.step()
        model.zero_grad(set_to_none=True)

        train_loss_f = train_loss.item()
        train_pc_loss_f = train_pc_loss.item()

        # Fast fail: abort if loss is exploding or NaN
        if math.isnan(train_loss_f) or train_loss_f > 100:
            print("FAIL")
            exit(1)

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
        mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / H100_BF16_PEAK_FLOPS
        remaining = max(0, TIME_BUDGET - total_training_time)

        print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | pc_loss: {train_pc_loss_f:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

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

    # Final eval
    model.eval()
    with autocast_ctx:
        val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

    # Final summary
    t_end = time.time()
    startup_time = t_start_training - t_start
    steady_state_mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10) / total_training_time / H100_BF16_PEAK_FLOPS if total_training_time > 0 else 0
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    ckpt_path = "checkpoint.pt"
    torch.save({"model": model.state_dict(), "config": asdict(config)}, ckpt_path)

    print("---")
    print(f"val_bpb:          {val_bpb:.6f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"mfu_percent:      {steady_state_mfu:.2f}")
    print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.1f}")
    print(f"depth:            {DEPTH}")
