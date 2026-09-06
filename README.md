# Attention-From-Scratch

Building the attention mechanism used in transformer models (GPT-style) from the ground up, in PyTorch — no `torch.nn.MultiheadAttention` shortcuts. Implemented step by step, from a plain dot-product attention to a batched, multi-head, causal self-attention module matching GPT-2's configuration.

## What's here

- **`code.ipynb`** — the full walkthrough, in order:
  1. **Basic attention** — computing attention scores via dot products, normalizing with softmax, and producing context vectors, first for a single query then for all tokens at once.
  2. **Self-attention with trainable weights** — introducing learnable `W_query`, `W_key`, `W_value` matrices (`SelfAttention_v1` using raw parameters, `SelfAttention_v2` using `nn.Linear`).
  3. **Causal attention** — masking out future tokens (so each position can only attend to itself and earlier tokens) via a triangular mask, plus dropout on attention weights and support for batched input (`CausalAttention`).
  4. **Multi-head attention** — first as a naive wrapper running several `CausalAttention` heads in parallel and concatenating results (`MultiHeadAttentionWrapper`), then as an efficient single-matrix implementation that splits Q/K/V into heads via reshaping instead of looping (`MultiHeadAttention`).
  5. A sanity check initializing `MultiHeadAttention` at GPT-2 (small) scale: `d_in = d_out = 768`, `num_heads = 12`, `context_length = 1024`.

- **`self_attention.py`** — the final, reusable `MultiHeadAttention` module extracted from the notebook, ready to import into other code.

## Usage

```python
import torch
from self_attention import MultiHeadAttention

d_in = d_out = 768
mha = MultiHeadAttention(
    d_in=d_in,
    d_out=d_out,
    context_length=1024,
    dropout=0.1,
    num_heads=12
)

x = torch.rand(2, 1024, d_in)   # (batch, tokens, embedding_dim)
context_vecs = mha(x)           # (2, 1024, 768)
```

## Requirements

- Python 3
- PyTorch (`pip install torch`)

## Reference

Based on the attention mechanism chapter from *Build a Large Language Model (From Scratch)* by Sebastian Raschka.
