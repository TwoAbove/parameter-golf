# Heterogeneous Data Ordering + LR Peak 2x

**val_bpb: TBD** (3-seed mean) | **~15.9 MB** | 8xH100 SXM, 600s | No TTT

**Improvement over current SOTA ([PR #1019](https://github.com/openai/parameter-golf/pull/1019), 1.1147 BPB):** TBD

## Results

| Seed | Steps | ms/step | Pre-quant BPB | **Sliding BPB** | Artifact |
|------|-------|---------|---------------|-----------------|----------|
| 42 | TBD | TBD | TBD | **TBD** | TBD |
| 314 | TBD | TBD | TBD | **TBD** | TBD |
| 999 | TBD | TBD | TBD | **TBD** | TBD |
| **Mean** | | | | **TBD** | |

---

## Main Changes

Built on [PR #1019](https://github.com/openai/parameter-golf/pull/1019) (AR Self-Gen GPTQ + XSA-all + BigramHash 3072x112). Two additions:

### 1. Heterogeneous Data Ordering (`DATA_ORDER=hetero`)

Training sequences are reordered so consecutive batches see maximally diverse content. The method:

1. **Per-token PCA scoring**: Fit PCA on bag-of-words token frequency vectors (100K subsample), project all sequences onto PC1 to get a single "topic score" per sequence.
2. **Interleave from extremes**: Sort by score, split in half, interleave — consecutive sequences come from opposite ends of the semantic spectrum.

This is done on-the-fly in `TokenStream._reorder()` at shard load time. Zero parameter cost, negligible compute overhead (~1s per shard).

### 2. LR Peak Overshoot (`LR_PEAK=2.0`)

During warmup, LR overshoots to 2x the base rate then decays to 1x during warmdown. This helps escape early local minima, but only when paired with hetero ordering (on uniform data it slightly hurts).

---

## Architecture

Identical to PR #1019:

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8 GQA heads, 4 KV heads) |
| MLP | 3x (1536) with LeakyReLU(0.5)^2 |
| Attention | XSA on all 11 layers |
| BigramHash | 3072 x dim=112 |
| RoPE | Partial (16/64 dims) |
| Optimizer | Parallel Muon (NS5) + Adam |
| Quantization | AR self-gen GPTQ int6 + LZMA9 |

## Reproduction

```bash
DATA_ORDER=hetero LR_PEAK=2.0 SEED=42 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```
