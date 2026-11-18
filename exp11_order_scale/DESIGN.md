# Exp 11 (planned): Compression predicts the path, not the endpoint: data order, Bayesianity and scale

**Status: draft pre-registration. Nothing has been run.**

## Motivation

- **Exp 08.** The same data was learnt in six orders. Prequential code length then predicted
  the generalisation of the *path ensemble* (Spearman ρ = 0.96), but not that of the final
  model (ρ = 0.21).
- **A recent result.** Prequential regret has been reported to track the final model's
  generalisation gap (Pearson > 0.9) on CIFAR ResNets, with a **fixed** data order
  (arXiv 2610.21376).
- **The Bayesian link.** For an exact Bayesian learner, prequential code length equals
  -log marginal likelihood, which is **order-invariant**. Excess code over a Bayesian
  reference equals the cumulative predictive KL (cf. arXiv 2507.11768, which studies this
  for in-context learning, not training).
- **What this suggests.** The spread of code length across orders of the same data measures
  how far SGD is from Bayesian inference.

## Questions

1. **Q1. Does the final-model link survive changes of order?** Re-run the fixed-order CIFAR
   result under varied orders.
2. **Q2. Does SGD become more Bayesian with scale?** Does the order spread of code length
   shrink with model size? Sutskever (2023) argues that large models approximate an ideal
   compressor, which predicts it should.

## Part A: CIFAR replication with order variation

- **Setup.** ResNets on CIFAR-10 (15k subsample), following arXiv 2610.21376.
- **Orders.** Random, easy-first, hard-first (by a reference model's loss), and class-blocked.
- **Measures.**
  - prequential regret per run;
  - generalisation gap of the final model;
  - generalisation gap of a checkpoint ensemble and of the weight average.
- **H-A1.** Under a fixed random order, across architectures, regret correlates with the
  final model's gap at Pearson >= 0.8. This replicates the original claim.
- **H-A2.** Across orders of the same architecture, the correlation with the final model's gap
  drops below 0.5, while the correlation with the checkpoint ensemble's gap stays >= 0.8.

## Part B: Order spread vs model size

- **Setup.**
  - Exp 05's four-source dataset (stories, facts, noise, dupes) and a TinyStories-only
    control.
  - GPT models at about 1M, 4M, 16M, 64M and 128M parameters, each with its own prior
    pretrained on the same tokens.
  - Six orders and 3 seeds per size.
- **Measures.**
  - Δ(N): the range of learnable code length across orders, divided by the mean.
  - Ensemble and final generalisation per order.
- **H-B1.** Δ(N) decreases monotonically with model size N.
- **H-B2.** log Δ is linear in log N, with R² >= 0.9 (a power law); report the exponent.
- **H-B3.** The Spearman correlation of code length with path-ensemble generalisation stays
  >= 0.8 at every size.

If H-B1 fails (the order gap does not shrink), that is reported as evidence against
"SGD ≈ ideal compressor" at these scales.

## Frozen settings for Part B (fixed before any run)

| Size | Layers | d_model | Heads | Parameters | Pretraining LR | Fine-tuning LR |
|---|---|---|---|---|---|---|
| m1 | 2 | 128 | 4 | 0.94M | 2e-3 | 6.7e-4 |
| m4 | 4 | 256 | 4 | 4.23M | 1.5e-3 | 5e-4 |
| m16 | 8 | 384 | 6 | 15.8M | 1e-3 | 3.3e-4 |
| m64 | 12 | 640 | 10 | 61.7M | 6e-4 | 2e-4 |
| m128 | 12 | 896 | 14 | 119.4M | 4e-4 | 1.3e-4 |

The fine-tuning LR is 5e-4 times the ratio of the size's pretraining LR to m4's.

- **Prior per size.** Every size is pretrained on the same 20M TinyStories tokens: one epoch,
  batch 32, cosine schedule, as in Exp 04. Its RHO reference model is that prior trained
  1,500 steps on holdout data.
- **Runs.** The six orders of Exp 05 x seeds 1-3, with the path ensemble on: 90 runs.
- **Order spread.** Δ(N) = (max − min) / mean, over the six orders, of the seed-mean learnable
  code length.
- **Not in Part B.** The TinyStories-only control and Part A are deferred.

## Compute and cost

About 20-40 GPU hours on a rented RTX 4090 or 5090: about $7-14 on a 4090, $14-28 on a 5090.

## Before running

- Read the full texts of arXiv 2610.21376, arXiv 2507.11768 and Requential Coding
  (NeurIPS 2026).
- Search for prior work on order sensitivity vs scale.
- Freeze hyperparameters on a dev seed.
