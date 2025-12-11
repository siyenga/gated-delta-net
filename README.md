# gated-delta-net
inference using the DNN
<img width="1024" height="1536" alt="image" src="https://github.com/user-attachments/assets/2ff60b6f-bacb-4e25-910e-58ab142a7414" />


                    ┌──────────────────────────────────────────┐
                    │              INPUT IMAGE                  │
                    │               (1 × 28 × 28)               │
                    └──────────────────────────────────────────┘
                                   │
                                   ▼
                     ┌────────────────────────┐
                     │     PATCHIFY (7×7)     │
                     │  16 patches, each 49d  │
                     └────────────────────────┘
                                   │
                     Tokens shape: (B, 16, 49)
                                   │
                                   ▼
         ┌──────────────────────────────────────────────────┐
         │         LINEAR TOKEN EMBEDDING (49 → d_model)    │
         └──────────────────────────────────────────────────┘
                                   │
                     Output: (B, 16, d_model)
                                   │
                                   ▼
                  ┌────────────────────────────┐
                  │  ADD POSITIONAL EMBEDDING  │
                  └────────────────────────────┘
                                   │
                                   ▼
             ┌─────────────────────────────────────────────┐
             │           STACK OF N GATED-DELTA BLOCKS     │
             │         (6 blocks in your configuration)     │
             └─────────────────────────────────────────────┘
                                   │
                                   ▼
                     ┌────────────────────────────┐
                     │    MEAN POOL (Tokens → 1)   │
                     │    Output: (B, d_model)     │
                     └────────────────────────────┘
                                   │
                                   ▼
                ┌─────────────────────────────────┐
                │     LAYER NORM + LINEAR HEAD    │
                │         (d_model → 10 classes)  │
                └─────────────────────────────────┘
                                   │
                                   ▼
                        ┌────────────────────┐
                        │    Predictions     │
                        │    (B, 10 logits)  │
                        └────────────────────┘

Input x  (B, T, d_model)
        │
        ├─► trunk = MLP(x)
        │       (B, T, d_model)
        │
        ├─► delta = MLP(x)
        │       (B, T, d_model)
        │
        └─► gate = sigmoid( Linear(x) )
                (B, T, d_model)

Gated Mix:
    mixed = gate * trunk + (1 - gate) * delta

Residual + Norm:
    out = LayerNorm( x + Dropout(mixed) )


                 ┌───────────────┐
                 │     INPUT x    │
                 └───────────────┘
        ┌───────────────────────────────────────────┐
        │ trunk MLP        delta MLP      gate proj │
        │  (d→4d→d)        (d→4d→d)        (d→d→sig)│
        └───────────────────────────────────────────┘
                 │             │                │
                 ▼             ▼                ▼
              trunk         delta              gate
                 │             │                │
                 └────── gated mixing ─────────┘
                           (elementwise)
                               │
                               ▼
                     residual + layernorm
                               │
                               ▼
                           OUTPUT
| Stage             | Shape            |
| ----------------- | ---------------- |
| Input image       | (1, 28, 28)      |
| Patchify 7×7      | 16 patches       |
| Tokens            | (B, 16, 49)      |
| Embedding         | (B, 16, d_model) |
| GatedDelta blocks | (B, 16, d_model) |
| Mean pool         | (B, d_model)     |
| Classifier        | (B, 10)          |




