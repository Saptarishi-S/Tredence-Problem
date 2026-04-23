# Sparse Gate Regularization on CIFAR-10: A Short Report

## Why L1 Penalty on Sigmoid Gates Encourages Sparsity

Each weight in the `PrunableLinear` layer is multiplied by a learned gate:

gate = sigmoid(gate_score)    ∈ (0, 1)
effective_weight = weight × gate

The sparsity loss added to the training objective is:

L_sparsity = mean(gate)   ≡   (1/N) Σ sigmoid(gate_score_i)

Minimizing this term is equivalent to an **L1-like penalty directly on the gate activations**. Here is why it drives sparsity:

- **Sigmoid output is always positive.** Unlike weights, which can cancel under L2 regularization, gates live in (0, 1). The only way to reduce their mean is to push individual values toward **0**, not toward some negative counterpart.
- **Gradient always points toward zero.** The gradient of `sigmoid(s)` with respect to `s` is `sigmoid(s) · (1 − sigmoid(s))`, which is strictly positive. So the sparsity loss always produces a negative gradient on every gate score, uniformly pulling all scores down — the optimizer must actively "fight" this pull to keep a gate open.
- **Effective pruning:** Once a gate falls below the threshold (0.1 in `compute_sparsity`), the corresponding weight contributes almost nothing to the forward pass. The network learns to keep gates open only when the associated weight is genuinely useful for classification.
- **λ controls the trade-off.** A larger λ amplifies this downward pull relative to the cross-entropy loss, so the network accepts a higher sparsity level in exchange for reduced classification pressure.

This is analogous to L1 regularization on weights (LASSO), which similarly shrinks coefficients to exactly zero — but here the squashing through sigmoid makes the sparsification effect particularly clean and bounded.

---

## Results Summary

Models were trained for 10 epochs on CIFAR-10 using a three-layer MLP (`3072 → 512 → 256 → 10`). The sparsity level is the percentage of gates below 0.1 at the end of training.

| Lambda (λ) | Test Accuracy (%) | Sparsity Level (%) |
|:----------:|:-----------------:|:------------------:|
| 0.1        | 56.40             | 13.70              |
| 0.5        | 56.62             | 45.22              |
| 2.0        | 57.07             | 78.99              |

### Key Observations

- **Sparsity scales strongly with λ.** Going from λ = 0.1 to λ = 2.0 raises the fraction of pruned gates from ~14% to ~79%, confirming that the L1-style penalty effectively drives gates toward zero.
- **Accuracy is remarkably stable.** Despite losing nearly 80% of its effective weights at λ = 2.0, the model matches — and marginally exceeds — the accuracy of the least-sparse model. This suggests the network contains substantial redundancy, and the regularization forces it to discover a more compact, efficient representation.
- **Best accuracy at highest sparsity (λ = 2.0, 57.07%).** The slight accuracy gain at higher λ is likely due to a mild regularization effect on the weights themselves (inactive gates prevent overfitting on noisy weight directions), though the differences are small enough to be within noise over a single run.
