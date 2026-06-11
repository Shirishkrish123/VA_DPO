# Variance-Aware Direct Preference Optimization (VA-DPO)

### Temporal Gradient-Norm Regularization for Stable Preference Alignment at Small Scales

This repository contains the codebase and experimental results for **VA-DPO (Variance-Aware Direct Preference Optimization)**, a temporal gradient-regularization method designed to stabilize preference optimization in small-scale language models (e.g., Qwen2-1.5B).

DPO training is known to exhibit high variance and instability when scaled down to smaller architectures. VA-DPO addresses this by adding a temporal gradient-norm smoothness penalty to the objective function, dampening sudden gradient spikes across steps:

$$\mathcal{L}_{\text{VA-DPO}}(t) = \mathcal{L}_{\text{DPO}}(t) + \lambda \left( \|g_t\| - \|g_{t-1}\| \right)^2$$

Where $g_t$ represents the unscaled gradient norm at step $t$, and $\lambda$ controls regularization strength.

---

## Key Empirical Results

All metrics below are computed over a 200-step training run using $N=20$ logged intervals. Standard deviations are calculated using the unbiased sample standard deviation estimator ($\text{ddof}=1$).

| Method | Mean Loss | Loss Std Dev | Mean Grad Norm | Grad Norm Std | Mean $\|\Delta g\|_t$ | Std $\|\Delta g\|_t$ | Reward Accuracy (Final / Mean) | Reward Margin (Final / Mean) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DPO Baseline** | 0.69079 | 0.00220 | 16.8622 | 4.7381 | 5.1168 | 5.0078 | 65.0% / 60.0% | 0.00555 / 0.00478 |
| **KTO Baseline** | 0.49967 | 0.00052 | 5.6067 | 1.5414 | *N/A* | *N/A* | *N/A* (Unpaired) | -0.00779 / 0.00065 |
| **VA-DPO ($\lambda=0.1$)** | 0.69084 | 0.00216 | 17.4090 | 4.0635 | 4.3381 | 3.7368 | 65.0% / 58.0% | 0.00419 / 0.00474 |
| **VA-DPO ($\lambda=0.2$)** | 0.69124 | 0.00193 | 17.1909 | 3.5454 | 3.6929 | 2.7120 | 50.0% / 58.0% | 0.00143 / 0.00391 |
| **VA-DPO ($\lambda=0.3$)** | 0.69103 | 0.00195 | 16.6123 | 3.3194 | 3.2535 | 2.0936 | 60.0% / 60.0% | 0.00389 / 0.00432 |

### Key Findings
1. **Successful Variance Damping:** As regularization strength $\lambda$ increases from $0.0$ to $0.3$, gradient norm variance drops by **30%** (Std dev $4.74 \rightarrow 3.32$) and step-to-step fluctuations $\|\Delta g\|_t$ drop by **58%** (Std dev $5.01 \rightarrow 2.09$).
2. **Conserved Downstream Alignment:** Downstream reward metrics (mean separation margin and accuracy) remain conserved, demonstrating that gradient smoothing stabilizes the optimization trajectory without degrading preference learning.

---

## Repository Structure

* `01_dataset_setup.py`: Downloads, caches, and shuffles the `trl-lib/ultrafeedback_binarized` dataset under seed 42.
* `02_train_dpo_baseline.py`: Runs baseline DPO training for 200 steps on 4-bit QLoRA.
* `03_train_kto.py`: Runs baseline KTO training for 200 steps on 4-bit QLoRA.
* `04_compute_gradient_variance.py`: Compiles results and generates comparative plots.
* `05_train_vadpo.py`: Core implementation of the VA-DPO trainer containing the custom `VADPOTrainer`.
* `06_kto_proxy_accuracy.py`: Auxiliary proxy evaluation scripts.

---

## Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone git@github.com:Shirishkrish23/VA_DPO.git
   cd VA_DPO
   ```

2. **Set up Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Running the Pipeline:**
   * **DPO Baseline:** `python 02_train_dpo_baseline.py`
   * **KTO Baseline:** `python 03_train_kto.py`
   * **VA-DPO:** `python 05_train_vadpo.py --lambda_va 0.1` (Accepts `--lambda_va` values like `0.1`, `0.2`, `0.3`)
   * **Stability Analysis:** `python 04_compute_gradient_variance.py`

---

## Key Implementation Highlights
* **FP16 GradScaler Correction:** In mixed-precision training, gradients are scaled by PyTorch to prevent underflow. VA-DPO handles this by dynamically reading `scaler.get_scale()` to adjust the computed gradient norms back to their unscaled physical magnitudes before applying the regularization penalty.
* **GPU-Asynchronous Computation:** All gradient norm calculations, differences, and penalty scaling are performed directly in GPU tensor space. This avoids CPU-GPU synchronization stalls (`.item()`), maintaining flat power draws and preventing physical hardware crashes.
