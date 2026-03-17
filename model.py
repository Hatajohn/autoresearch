"""
GPT model and components for autoresearch pretraining.
Single-file model definition: config, attention, MLP, blocks, PC heads.
Flash Attention: set model.fa3 from train.py after loading kernels (optional).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Injected by train.py after loading kernels (or left None for SDPA fallback).
fa3 = None


# ---------------------------------------------------------------------------
# Config and helpers
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
    # Run-time scalars (buffers); set by train.py so sweeping them doesn't trigger recompile.
    pc_focal_gamma: float = 1.0
    kl_weight: float = 0.01
    logit_softcap: float = 15.0
    aux_horizons: tuple = ((2, 0.15), (4, 0.05))
    use_stochastic_layers: bool = True   # False = all layers use _NoStoch (no KL, no noise)


# Default aux horizons (used by smoke_test and for default config).
AUX_HORIZONS = ((2, 0.15), (4, 0.05))


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """True if layer should have Value Embedding (alternating, last always included)."""
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
    i = torch.arange(T, device=device).unsqueeze(1)
    j = torch.arange(T, device=device).unsqueeze(0)
    causal = j > i
    if window > 0 and window < T:
        causal = causal | ((i - j) > window)
    return torch.zeros(T, T, device=device, dtype=dtype).masked_fill(causal, float('-inf'))


# ---------------------------------------------------------------------------
# Attention, MLP, block
# ---------------------------------------------------------------------------

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
        self.attn_temp = nn.Parameter(torch.ones(1))

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
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
    """Reparameterized sampling: q(z|x)=N(mu, sigma²); train=sampled z, eval=mu. Returns (z, kl)."""
    def __init__(self, n_embd):
        super().__init__()
        self.mu_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.log_sigma_proj = nn.Linear(n_embd, n_embd, bias=True)
        self.register_buffer("noise_scale", torch.ones(1))

    def train(self, mode=True):
        super().train(mode)
        self.noise_scale.fill_(1.0 if mode else 0.0)
        return self

    def forward(self, x):
        mu = self.mu_proj(x)
        log_sigma = self.log_sigma_proj(x).clamp(-6.0, 2.0)
        sigma = log_sigma.exp()
        z = mu + self.noise_scale.to(dtype=mu.dtype) * sigma * torch.randn_like(mu)
        kl = 0.5 * (sigma.float().pow(2) + mu.float().pow(2)
                    - 2.0 * log_sigma.float() - 1.0).mean()
        return z, kl


class TemporalSummarizer(nn.Module):
    """Causal depthwise conv + proj; residual. Keeps T constant for fullgraph compile."""
    def __init__(self, n_embd, stride=4):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv1d(n_embd, n_embd, kernel_size=stride,
                              padding=stride - 1, groups=n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.size()
        h = x.transpose(1, 2)
        h = self.conv(h)[..., :T]
        h = h.transpose(1, 2)
        h = self.proj(h)
        return x + h


class _NoVE(nn.Module):
    """Sentinel for non-VE layers; CausalSelfAttention skips value residual when None."""
    def forward(self, idx):
        return None


class _NoStoch(nn.Module):
    """Sentinel for non-stochastic layers; returns (x, zero) so forward has no branch."""
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


# ---------------------------------------------------------------------------
# GPT
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        horizons = getattr(config, 'aux_horizons', AUX_HORIZONS)
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        _summarizer_layers = {max(1, config.n_layer // 3), 2 * max(1, config.n_layer // 3)}
        self.summarizers = nn.ModuleList([
            TemporalSummarizer(config.n_embd) if i in _summarizer_layers else nn.Identity()
            for i in range(config.n_layer)
        ])
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.aux_lm_heads = nn.ModuleList([
            nn.Linear(config.n_embd, config.vocab_size, bias=False)
            for _ in horizons
        ])
        self.lm_head_log_scale = nn.Parameter(torch.zeros(()))
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleList([
            nn.Embedding(config.vocab_size, kv_dim) if has_ve(i, config.n_layer) else _NoVE()
            for i in range(config.n_layer)
        ])
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        pc_gamma = getattr(config, 'pc_focal_gamma', 1.0)
        kl_w = getattr(config, 'kl_weight', 0.01)
        self.register_buffer("pc_focal_gamma", torch.tensor(pc_gamma, dtype=torch.bfloat16), persistent=False)
        self.register_buffer("kl_weight", torch.tensor(kl_w, dtype=torch.bfloat16), persistent=False)
        self.register_buffer("pc_scale", torch.tensor(config.n_embd ** 0.5, dtype=torch.bfloat16), persistent=False)
        _stochastic_indices = set(range(2, config.n_layer, 3)) if getattr(config, 'use_stochastic_layers', True) else set()
        self.stochastic_layers = nn.ModuleList([
            StochasticLayer(config.n_embd) if i in _stochastic_indices else _NoStoch()
            for i in range(config.n_layer)
        ])
        self.pc_log_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        self.pc_fc_w = nn.Parameter(torch.zeros(config.n_layer, config.pc_head_dim, config.n_embd))
        self.pc_proj_w = nn.Parameter(torch.zeros(config.n_layer, config.pc_head_dim, config.n_embd))
        self.pc_gate_w = nn.Parameter(torch.zeros(config.n_layer, config.n_embd))
        self.pc_gate_b = nn.Parameter(torch.ones(config.n_layer))
        self.register_buffer('pc_ema', torch.zeros(config.n_layer - 1, config.n_embd))

    @torch.no_grad()
    def init_weights(self):
        cfg = self.config
        pc_gamma = getattr(cfg, 'pc_focal_gamma', 1.0)
        kl_w = getattr(cfg, 'kl_weight', 0.01)
        wte_std = 1.75 / (cfg.n_embd ** 0.5)
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=wte_std)
        for head in self.aux_lm_heads:
            torch.nn.init.normal_(head.weight, mean=0.0, std=0.001)
        n_embd = cfg.n_embd
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
        torch.nn.init.zeros_(self.pc_fc_w)
        torch.nn.init.zeros_(self.pc_proj_w)
        torch.nn.init.zeros_(self.pc_gate_w)
        self.pc_gate_b.fill_(1.0)
        self.resid_lambdas.fill_(math.log(math.e - 1))
        self.pc_log_lambdas.fill_(0.0)
        for ve in self.value_embeds:
            if isinstance(ve, nn.Embedding):
                torch.nn.init.uniform_(ve.weight, -s, s)
        for sl in self.stochastic_layers:
            if isinstance(sl, StochasticLayer):
                torch.nn.init.eye_(sl.mu_proj.weight)
                torch.nn.init.zeros_(sl.log_sigma_proj.weight)
                torch.nn.init.constant_(sl.log_sigma_proj.bias, -3.0)
            else:
                sl._zero.zero_()
        for summ in self.summarizers:
            if isinstance(summ, TemporalSummarizer):
                torch.nn.init.zeros_(summ.proj.weight)
                torch.nn.init.ones_(summ.conv.weight)
        head_dim = cfg.n_embd // cfg.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds:
            if isinstance(ve, nn.Embedding):
                ve.to(dtype=torch.bfloat16)
        self.pc_focal_gamma.fill_(pc_gamma)
        self.kl_weight.fill_(kl_w)
        self.pc_scale.fill_(float(cfg.n_embd) ** 0.5)
        self.lm_head_log_scale.fill_(0.0)
        self.pc_ema.zero_()
        for sl in self.stochastic_layers:
            if isinstance(sl, StochasticLayer):
                sl.noise_scale.fill_(1.0)

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
            short_window = long_window // 2
            window_sizes = [(short_window if i < n // 2 else long_window, 0) for i in range(n)]
            window_sizes[-1] = (long_window, 0)
            return window_sizes
        if pattern == "LOG":
            tile = 128
            if config.log_min_window > 0:
                v = config.log_min_window
                assert v >= 16 and (v & (v - 1)) == 0, f"log_min_window must be power of 2 and >= 16, got {v}"
                min_w, tile_eff = v, min(tile, v)
            else:
                min_w = max(64, long_window // n)
                min_w = ((min_w + tile - 1) // tile) * tile
                tile_eff = tile
            window_sizes = []
            for i in range(n - 1):
                t = i / (n - 1)
                w = min_w * (long_window / min_w) ** t
                w = max(tile_eff, round(w / tile_eff) * tile_eff)
                window_sizes.append((w, 0))
            window_sizes.append((long_window, 0))
            return window_sizes
        assert all(c in "SL" for c in pattern)
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = [char_to_window[pattern[i % len(pattern)]] for i in range(n)]
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds if isinstance(ve, nn.Embedding))
        wte_numel = self.transformer.wte.weight.numel()
        lm_head_numel = 0 if self.lm_head.weight is self.transformer.wte.weight else self.lm_head.weight.numel()
        nparams_exclude = wte_numel + lm_head_numel + value_embeds_numel
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        attn_flops = sum(12 * h * q * (t if w[0] < 0 else min(w[0], t)) for w in self.window_sizes)
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
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
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head, 'lm_head_tied': lm_head_tied,
            'aux_heads': aux_heads, 'transformer_matrices': transformer_matrices, 'scalars': scalars,
            'summarizers': summarizers, 'stochastic': stochastic, 'pc_heads': pc_heads, 'total': total,
        }

    @torch.no_grad()
    def get_pc_diagnostics(self):
        n = self.config.n_layer
        pc_weights = F.softplus(self.pc_log_lambdas).tolist()
        resid = F.softplus(self.resid_lambdas).tolist()
        gate_biases = self.pc_gate_b.tolist()
        attn_temps = [(b.attn.attn_temp.square() + 0.01).item() for b in self.transformer.h]
        sigma_bias_exp = [sl.log_sigma_proj.bias.exp().mean().item()
                          for sl in self.stochastic_layers if isinstance(sl, StochasticLayer)]
        return {"pc_weights": pc_weights, "resid": resid, "gate_biases": gate_biases,
                "attn_temps": attn_temps, "sigma_bias_exp": sigma_bias_exp, "n_layer": n}

    def _get_optimizer_param_lists(self):
        pc_head_params = [self.pc_fc_w, self.pc_proj_w, self.pc_gate_w, self.pc_gate_b]
        pc_head_param_ids = {id(p) for p in pc_head_params}
        attn_temp_params = [block.attn.attn_temp for block in self.transformer.h]
        attn_temp_param_ids = {id(p) for p in attn_temp_params}
        exclude_ids = pc_head_param_ids | attn_temp_param_ids
        matrix_params = [p for p in self.transformer.h.parameters() if id(p) not in exclude_ids]
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        aux_head_params = list(self.aux_lm_heads.parameters())
        summarizer_params = list(self.summarizers.parameters())
        stochastic_params = list(self.stochastic_layers.parameters())
        resid_params = [self.resid_lambdas, self.pc_log_lambdas, self.lm_head_log_scale] + attn_temp_params
        result = [
            aux_head_params, embedding_params, value_embeds_params, resid_params,
            pc_head_params, summarizer_params, stochastic_params,
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            result.append([p for p in matrix_params if p.shape == shape])
        return result

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5, stoch_lr_scale=0.1):
        from optimizer import MuonAdamW
        model_dim = self.config.n_embd
        param_lists = self._get_optimizer_param_lists()
        (aux_head_params, embedding_params, value_embeds_params, resid_params,
         pc_head_params, summarizer_params, stochastic_params) = param_lists[:7]
        matrix_param_lists = param_lists[7:]
        matrix_params = [p for lst in matrix_param_lists for p in lst]
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(aux_head_params) + len(value_embeds_params) + len(resid_params) +
            len(pc_head_params) + len(summarizer_params) + len(stochastic_params))
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind='adamw', params=aux_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=pc_head_params, lr=matrix_lr * 0.1, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=summarizer_params, lr=matrix_lr, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=stochastic_params, lr=matrix_lr * stoch_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        ]
        for group_params in matrix_param_lists:
            param_groups.append(dict(kind='muon', params=group_params, lr=matrix_lr,
                                     momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        softcap = getattr(self.config, 'logit_softcap', 15.0)
        horizons = getattr(self.config, 'aux_horizons', AUX_HORIZONS)

        x = self.transformer.wte(idx)
        x = norm(x)
        kl_loss = torch.zeros((), dtype=torch.float32, device=x.device)
        resid_scales = F.softplus(self.resid_lambdas).unbind(0)

        layer_outs = []
        for i, block in enumerate(self.transformer.h):
            x = resid_scales[i] * x
            x = self.summarizers[i](x)
            ve = self.value_embeds[i](idx)
            x = block(x, ve, cos_sin, self.window_sizes[i])
            x, kl_contrib = self.stochastic_layers[i](x)
            kl_loss = kl_loss + kl_contrib
            layer_outs.append(x)

        L = self.config.n_layer - 1
        normed_outs = [norm(layer_outs[i]) for i in range(self.config.n_layer)]
        x = normed_outs[-1]

        logits = self.lm_head(x) * self.lm_head_log_scale.exp()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            token_ce = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                ignore_index=-1, reduction='none',
            ).view(B, T)
            if reduction == 'mean':
                h_stack = torch.stack(normed_outs[:L])
                ema_target = self.pc_ema.to(dtype=h_stack.dtype).unsqueeze(1).unsqueeze(1).expand_as(h_stack)
                fc_out = torch.einsum('lbtc,ldc->lbtd', h_stack, self.pc_fc_w[:L])
                pred = h_stack + torch.einsum('lbtd,ldc->lbtc', torch.tanh(fc_out), self.pc_proj_w[:L])
                raw_error = (ema_target - pred) / self.pc_scale
                pc_w = F.softplus(self.pc_log_lambdas[:L]).view(L, 1, 1)
                token_pc = pc_w * raw_error.pow(2).mean(dim=-1)
                gate_logit = torch.einsum('lbtc,lc->lbt', h_stack.detach(), self.pc_gate_w[:L])
                gate = torch.sigmoid(gate_logit + self.pc_gate_b[:L].view(L, 1, 1))
                pc_loss_map = (gate * token_pc).sum(dim=0)
                loss = token_ce.mean()
                for head, (horizon_k, horizon_w) in zip(self.aux_lm_heads, horizons):
                    aux_logits = head(x[:, :-horizon_k])
                    aux = F.cross_entropy(
                        aux_logits.contiguous().view(-1, aux_logits.size(-1)),
                        targets[:, horizon_k:].contiguous().view(-1),
                        ignore_index=-1, reduction='mean',
                    )
                    loss = loss + horizon_w * aux
                tc_det = token_ce.detach()
                difficulty = (tc_det / (tc_det.mean() + 1e-6)).pow(self.pc_focal_gamma)
                pc_loss = (pc_loss_map * difficulty).mean() / max(1, len(self.transformer.h) - 1)
                aux_loss = pc_loss + self.kl_weight * kl_loss
                layer_means = torch.stack(
                    [normed_outs[i + 1].detach().float().mean(dim=(0, 1)) for i in range(L)]
                )
                return loss, aux_loss, layer_means
            else:
                loss = token_ce
                aux_loss = torch.zeros((), dtype=torch.float32, device=x.device) + self.kl_weight * kl_loss
                return loss, aux_loss
        return logits
