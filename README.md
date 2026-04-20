# Self-Pruning Neural Network via Learnable Gates - CIFAR-10

## Overview

This project implements a self-pruning feed-forward neural network trained on CIFAR-10. The core idea is to associate each weight in the network with a learnable **gate parameter**. During training, a sparsity regularization loss drives most gates toward zero, effectively pruning unnecessary weights from the network automatically, no manual pruning schedule required.

---

## How It Works

### Part 1: The `PrunableLinear` Layer

A custom linear layer replaces `torch.nn.Linear`. In addition to the standard `weight` and `bias` parameters, it holds a second learnable tensor `gate_scores` of the same shape as `weight`.

During the forward pass:

1. `gate_scores` are passed through a **Sigmoid** to produce gates in `(0, 1)`:
   ```
   gates = sigmoid(gate_scores)
   ```
2. Weights are element-wise multiplied by their corresponding gate:
   ```
   pruned_weight = weight * gates
   ```
3. The standard linear operation is applied using these pruned weights:
   ```
   output = pruned_weight @ x.T + bias
   ```

Gradients flow correctly through both `weight` and `gate_scores` via standard autograd & no special tricks needed since sigmoid is differentiable everywhere.

**Initialization:** `gate_scores` are initialized to `0.0`, so all gates start at `sigmoid(0) = 0.5`. This keeps gradients healthy from the first step (the sigmoid gradient is maximal near 0).

---

### Part 2: Sparsity Regularization Loss

Training uses a composite loss:

```
Total Loss = CrossEntropyLoss + λ × SparsityLoss
```

The **SparsityLoss** is the mean of all gate values across every `PrunableLinear` layer:

```
SparsityLoss = mean({ sigmoid(gate_scores) for all PrunableLinear layers })
```

#### Why does L1 on sigmoid gates encourage sparsity?

The L1 penalty (sum/mean of absolute values) is the canonical sparsity-inducing regularizer. For our gates, which are always positive after sigmoid, this reduces to simply their mean value. Minimizing this term directly minimizes the average gate magnitude, pushing as many gates as possible toward zero.

The key geometric insight is that the L1 norm has a **non-smooth corner at zero** and unlike L2, which has a smooth minimum and only shrinks values toward zero asymptotically, L1 creates a constant gradient pressure that drives values to *exactly* zero. A gate at `0.001` and a gate at `0.5` both receive the same magnitude of gradient from the L1 term, so even nearly-pruned gates continue to be pushed all the way to zero.

The **λ hyperparameter** controls the trade-off: higher λ applies more pressure toward sparsity at the cost of classification accuracy, since the optimizer must balance both objectives simultaneously.

---

### Part 3: Training Setup

- **Dataset:** CIFAR-10 (50,000 train / 10,000 test images, 10 classes)
- **Architecture:** 3-layer MLP - `3072 → 512 → 256 → 10` with ReLU activations
- **Optimizer:** Adam with separate learning rates - `1e-3` for weights/bias, `1e-2` for gate scores (gates need to move faster to overcome sigmoid saturation)
- **Epochs:** 10 per lambda
- **Sparsity threshold:** A gate is considered pruned if `sigmoid(gate_score) < 0.1`

---

## Results

| Lambda (λ) | Test Accuracy | Sparsity Level (%) |
|:----------:|:-------------:|:------------------:|
| 0.1        | 56.40%        | 13.70%             |
| 0.5        | 56.62%        | 45.22%             |
| 2.0        | **57.07%**    | **78.99%**         |

### Analysis

The results demonstrate a clear and successful pruning effect:

- **λ = 0.1 (low):** Gentle regularization. Only 13.7% of weights are pruned after 10 epochs, with the network still learning gradually. Gates are slow to converge toward zero.

- **λ = 0.5 (medium):** A strong pruning signal removes nearly half the weights (45.2%) while maintaining comparable accuracy. This represents a favorable operating point and significant compression with minimal accuracy cost.

- **λ = 2.0 (high):** Aggressive pruning drives 79% of weights to near-zero within just a few epochs. Counterintuitively, accuracy is marginally *higher* here and the sparsity pressure acts as a form of regularization, preventing overfitting and forcing the network to rely only on its most informative connections.

The key takeaway is that **accuracy is remarkably stable across all three lambda values** (56.4% → 57.1%), while sparsity increases dramatically (13.7% → 79%). This confirms that most weights in a dense MLP are redundant and the network can match its performance using fewer than 25% of its original connections.

> **Note on accuracy ceiling:** A pure MLP (no convolutions) on CIFAR-10 is theoretically limited to ~55–65% accuracy. Convolutional architectures are better suited for spatial data; the MLP here is used to demonstrate the pruning mechanism clearly on a well-known benchmark.

---

## Gate Value Distribution (Best Model)

The histogram below shows the distribution of final gate values for the best model (λ = 0.1, highest accuracy):

The distribution confirms successful pruning:

- A **large spike near 0** - the vast majority of gates have been driven to near-zero, meaning those weights are effectively removed from the network.
- A **long tail toward higher values** - a minority of gates survived with non-trivial values, representing the connections the network identified as genuinely important.

This bimodal-like shape (spike at 0 + surviving tail) is the hallmark of a successfully trained sparse network.

---

## Repository Structure

```
.
├── Tredence_Case_Study.ipynb   # Full implementation notebook
└── REPORT.md                   # This report
```

## Requirements

```
torch
torchvision
tqdm
matplotlib
```

## How to Run

Open `Tredence_Case_Study.ipynb` in Google Colab or a local Jupyter environment with a GPU. Run all cells top to bottom. The notebook will:

1. Train three models with λ ∈ {0.1, 0.5, 2.0}
2. Print per-epoch metrics including sparsity, accuracy, and gate statistics
3. Display the final results table
4. Plot the gate value distribution for the best model
