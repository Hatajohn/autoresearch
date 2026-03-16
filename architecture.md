# Model Architecture — pc-architecture-v2

> Config: `n_layer=8, n_embd=512, n_head=4, n_kv_head=4, head_dim=128, seq_len=2048`

---

## Data Flow

```mermaid
%%{init: {"theme": "dark"}}%%
flowchart TD
    TOK["tokens  B x T"]
    EMB["Embedding  vocab -> 512  bf16"]
    N0["RMSNorm"]
    TOK --> EMB --> N0

    subgraph LAYER["One layer  repeated x8  i = 0...7"]
        SCALE["resid_lambdas[i] * x"]
        SUMM["TemporalSummarizer  layers 2 and 4 only\nConv1d causal  + Linear  residual add"]
        ATTN["CausalSelfAttention\nRoPE   QK-norm   windowed FA3\nattn_temp^2 precision   value embeds on alt layers"]
        MLP_N["MLP   512 -> 2048 -> 512\nReLU^2 activation"]
        STOCH["StochasticLayer  layers 2 and 5 only\nz = mu + noise * sigma   adds KL loss"]
        layer_outs["layer_outs[i] = x\ncollect all 8 outputs"]

        SCALE --> SUMM --> ATTN --> MLP_N
        MLP_N --> STOCH --> layer_outs
    end

    subgraph PHASE2["Phase 2  broadcast PC  after all layers"]
        BROADCAST["broadcast = norm(layer_outs[-1]).detach()\nfinal layer is the global top-down target"]
        PRED["PredHead(norm(layer_outs[i]))\npredict broadcast from each layer 0..n-2"]
        ERR["raw_error = (broadcast - pred) / pc_scale"]
        ELBO["ELBO  L^2 * err^2 - log(L^2)"]
        GATE["routing_gate  sigmoid gate  (B,T)\nweights which tokens get strong PC signal"]
        PCLOSS_A["pc_loss_map += gate * ELBO\ngradient flows back to each layer's params"]
        BROADCAST --> PRED --> ERR --> ELBO --> PCLOSS_A
        ERR --> GATE --> PCLOSS_A
    end

    N0 --> SCALE
    layer_outs --> BROADCAST
    layer_outs --> OUTNORM

    OUTNORM["RMSNorm"]
    HEAD["lm_head   Linear 512 -> vocab   no bias"]
    CAP["softcap   15 * tanh(logits / 15)"]
    OUTNORM --> HEAD --> CAP

    subgraph LOSS["Loss  training only"]
        CE["cross-entropy per token  B x T"]
        AUX["+ 0.15 * CE t+2   + 0.05 * CE t+4\nmulti-timescale auxiliary"]
        FOCAL["difficulty = (token_ce / mean)^gamma\nfocal weight  hard tokens get more PC signal"]
        PCLOSS["pc_loss = mean(loss_map * difficulty) / 8\n+ kl_weight * kl_loss"]
        TOTAL["total = main_loss + PC_WEIGHT * aux_loss"]
        CE --> AUX --> TOTAL
        CE --> FOCAL --> PCLOSS --> TOTAL
    end

    CAP --> CE

    classDef buf   fill:#1e293b,stroke:#64748b,color:#94a3b8,stroke-width:1px
    classDef inp   fill:#1e3a8a,stroke:#60a5fa,color:#e0f2fe,stroke-width:2px
    classDef norm  fill:#312e81,stroke:#818cf8,color:#e0e7ff,stroke-width:1px
    classDef scale fill:#3b0764,stroke:#a855f7,color:#f3e8ff,stroke-width:1px
    classDef summ  fill:#164e63,stroke:#22d3ee,color:#cffafe,stroke-width:2px
    classDef attn  fill:#4c1d95,stroke:#a78bfa,color:#ede9fe,stroke-width:2px
    classDef mlp   fill:#14532d,stroke:#4ade80,color:#dcfce7,stroke-width:2px
    classDef pc    fill:#78350f,stroke:#fbbf24,color:#fef3c7,stroke-width:2px
    classDef stoch fill:#881337,stroke:#fb7185,color:#ffe4e6,stroke-width:2px
    classDef hist  fill:#1e293b,stroke:#475569,color:#cbd5e1,stroke-width:1px
    classDef out   fill:#1e3a5f,stroke:#3b82f6,color:#dbeafe,stroke-width:2px
    classDef loss  fill:#422006,stroke:#f97316,color:#ffedd5,stroke-width:2px
    classDef total fill:#713f12,stroke:#fbbf24,color:#fef9c3,stroke-width:3px

    class TOK,EMB        inp
    class N0,OUTNORM     norm
    class SCALE          scale
    class SUMM           summ
    class ATTN           attn
    class MLP_N          mlp
    class layer_outs  buf
    class BROADCAST,PRED,ERR,ELBO,GATE,PCLOSS_A  pc
    class STOCH          stoch
    class HIST           hist
    class HEAD,CAP       out
    class CE,AUX,FOCAL,PCLOSS  loss
    class TOTAL          total
```

---

## Attention Windows (seq=2048, n=8)

Three available patterns — window size per layer:

```mermaid
%%{init: {'theme': 'dark'}}%%
xychart-beta
    title "LOG (default) - geometric ~1.3x per layer, 8 kernels"
    x-axis ["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7"]
    y-axis "Window (tokens)" 0 --> 2200
    bar  [256, 384, 512, 640, 896, 1152, 1536, 2048]
```

```mermaid
%%{init: {'theme': 'dark'}}%%
xychart-beta
    title "PROGRESSIVE - step at n//2, 2 kernels"
    x-axis ["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7"]
    y-axis "Window (tokens)" 0 --> 2200
    bar  [1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048]
```

```mermaid
%%{init: {'theme': 'dark'}}%%
xychart-beta
    title "SSSL (original) - alternating, 2 kernels"
    x-axis ["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7"]
    y-axis "Window (tokens)" 0 --> 2200
    bar  [1024, 1024, 1024, 2048, 1024, 1024, 1024, 2048]
```

| Pattern | Min window | Scaling | Unique kernels | Default |
|---------|-----------|---------|----------------|---------|
| `LOG` | 256 | ~1.3× per layer (geometric) | 8 | **yes** |
| `PROGRESSIVE` | 1024 | Step at n//2 | 2 | — |
| `SSSL` | 1024 | Alternating | 2 | — |

`LOG` starts narrow (256 = seq/8) so early layers are forced to extract local features, then expands geometrically. Each layer sees ~1.3× the context of the previous, matching the DTM timescale hierarchy most closely. The 8 unique FA3 kernels are compiled once and cached to disk; subsequent runs pay no extra startup cost.

---

## Broadcast PC Targets

```mermaid
%%{init: {"theme": "dark"}}%%
flowchart LR
    L0["L0"]
    L1["L1"]
    L2["L2"]
    L3["L3"]
    L4["L4"]
    L5["L5"]
    L6["L6"]
    L7["L7  broadcast target"]

    L0 -->|predict| L7
    L1 -->|predict| L7
    L2 -->|predict| L7
    L3 -->|predict| L7
    L4 -->|predict| L7
    L5 -->|predict| L7
    L6 -->|predict| L7

    classDef pred   fill:#78350f,stroke:#fbbf24,color:#fef3c7,stroke-width:2px
    classDef target fill:#4c1d95,stroke:#a78bfa,color:#ede9fe,stroke-width:3px

    class L0,L1,L2,L3,L4,L5,L6  pred
    class L7  target
```

All layers 0..n-2 predict the final layer's normalised output (broadcast target, shown in purple). This is the most faithful implementation of hierarchical predictive coding: the highest level of the hierarchy sets a global top-down prediction that every lower level must learn to match. Layer 7 is detached — it sets the context without receiving gradient through the PC path.

---

## Special Layers at a Glance

| Layer | TemporalSummarizer | Value Embed | StochasticLayer | Skip dist |
|-------|--------------------|-------------|-----------------|-----------|
| 0 | — | — | — | 1 |
| 1 | — | ✓ | — | 1 |
| 2 | ✓ | — | ✓ | 2 |
| 3 | — | ✓ | — | 2 |
| 4 | ✓ | — | — | 3 |
| 5 | — | ✓ | ✓ | 3 |
| 6 | — | — | — | 4 |
| 7 | — | ✓ | — | 4 |

---

## Parameter Groups and Optimizers

| Group | Parameters | Optimizer | LR |
|-------|-----------|-----------|-----|
| `wte` | vocab × 512 | AdamW | `embedding_lr / √(512/768)` |
| `lm_head` | vocab × 512 | AdamW | `unembedding_lr / √(512/768)` |
| `value_embeds` (×4) | 4 × vocab × 512 | AdamW | `embedding_lr / √(512/768)` |
| `resid_lambdas`, `pc_lambdas`, `attn_temp` (24 scalars) | 24 | AdamW | `scalar_lr × 0.01` |
| `pred_head` + `routing_gate` (×8) | ~4.2 M | AdamW | `matrix_lr` |
| `TemporalSummarizer` (×2) | ~530 K | AdamW | `matrix_lr` |
| `StochasticLayer` (×2) | ~1 M | AdamW | `matrix_lr` |
| All other backbone matrices | ~29 M | **Muon** | `matrix_lr` |

LR schedule: flat → warmdown over last 50% of budget, cosine to 0.

---

## Sweepable Buffers (no recompile when changed)

| Buffer | Default | Effect |
|--------|---------|--------|
| `PC_WEIGHT` | 0.1 | Scales total `aux_loss` contribution to the gradient |
| `pc_alpha` | 0.1 | Scales the top-down routing correction added back to `x` |
| `pc_focal_gamma` | 1.0 | Exponent for output-difficulty weighting of `pc_loss` (0 = uniform) |
| `kl_weight` | 0.01 | Scales the KL divergence loss from stochastic layers |
| `pc_scale` | √512 ≈ 22.6 | Fixed denominator making prediction errors dimensionless (Bogacz 2017) |
