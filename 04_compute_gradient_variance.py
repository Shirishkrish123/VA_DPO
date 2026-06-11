"""
04_compute_gradient_variance.py
Compute quantitative stability metrics from DPO, KTO, and VA-DPO trainer logs.
Outputs comparison tables and bar plots for the paper.
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt

EXPERIMENTS = {
    "DPO": "outputs/dpo_baseline/checkpoint-200/trainer_state.json",
    "KTO": "outputs/kto/checkpoint-200/trainer_state.json",
    # VA-DPO entries will be added after training
    # "VA-DPO (0.01)": "outputs/vadpo_lambda_0.01/checkpoint-200/trainer_state.json",
}

OUTPUT_DIR = "./outputs/stability_analysis"

def load_log_history(path):
    """Load trainer_state.json and return the log_history list."""
    with open(path) as f:
        data = json.load(f)
    return data["log_history"]

def compute_stability_metrics(logs, method_name):
    """Compute all stability metrics for a single method."""
    grad_norms = [e["grad_norm"] for e in logs if "grad_norm" in e]
    losses = [e["loss"] for e in logs if "loss" in e]
    margins = [e["rewards/margins"] for e in logs if "rewards/margins" in e]
    
    # Temporal differences (step-to-step changes)
    grad_diffs = np.diff(grad_norms)
    loss_diffs = np.diff(losses)
    
    metrics = {
        "method": method_name,
        # Gradient stability
        "grad_norm_mean": float(np.mean(grad_norms)),
        "grad_norm_std": float(np.std(grad_norms, ddof=1)),
        "grad_norm_var": float(np.var(grad_norms, ddof=1)),
        "grad_norm_max": float(np.max(grad_norms)),
        "grad_norm_min": float(np.min(grad_norms)),
        "grad_norm_range": float(np.max(grad_norms) - np.min(grad_norms)),
        "max_grad_spike": float(np.max(np.abs(grad_diffs))),
        # Temporal smoothness (the VA-DPO regularizer target)
        "grad_temporal_variance": float(np.mean(grad_diffs**2)),
        # Loss stability
        "loss_std": float(np.std(losses, ddof=1)),
        "loss_var": float(np.var(losses, ddof=1)),
        "loss_temporal_variance": float(np.mean(loss_diffs**2)),
        # Reward separation
        "margin_mean": float(np.mean(margins)),
        "margin_std": float(np.std(margins, ddof=1)),
        "margin_final": float(margins[-1]) if margins else 0.0,
    }
    return metrics

def print_comparison_table(all_metrics):
    """Print a formatted comparison table."""
    headers = [
        "Metric", 
        *[m["method"] for m in all_metrics]
    ]
    
    rows = [
        ("Grad Norm Mean", "grad_norm_mean"),
        ("Grad Norm Std", "grad_norm_std"),
        ("Grad Norm Range", "grad_norm_range"),
        ("Max Grad Spike", "max_grad_spike"),
        ("Grad Temporal Var", "grad_temporal_variance"),
        ("Loss Std", "loss_std"),
        ("Loss Temporal Var", "loss_temporal_variance"),
        ("Margin Mean", "margin_mean"),
        ("Margin Final", "margin_final"),
    ]
    
    # Print header
    header_fmt = "{:<22}" + "{:>15}" * len(all_metrics)
    print("\n" + "=" * (22 + 15 * len(all_metrics)))
    print(header_fmt.format(*headers))
    print("=" * (22 + 15 * len(all_metrics)))
    
    for label, key in rows:
        values = [f"{m[key]:.6f}" for m in all_metrics]
        print(header_fmt.format(label, *values))
    
    print("=" * (22 + 15 * len(all_metrics)))

def generate_bar_plots(all_metrics):
    """Generate comparative bar plots."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    methods = [m["method"] for m in all_metrics]
    colors = {
        "DPO": "#e74c3c",          # Crimson Red
        "KTO": "#3498db",          # Blue
        "VA-DPO (0.1)": "#85e3b2",  # Light Emerald
        "VA-DPO (0.2)": "#2ecc71",  # Medium Emerald
        "VA-DPO (0.3)": "#1e8449",  # Dark Emerald
    }
    
    plot_configs = [
        ("grad_norm_std", "Gradient Norm Std Dev", "Stability (lower = more stable)"),
        ("grad_temporal_variance", "Gradient Temporal Variance", "Temporal Smoothness (lower = smoother)"),
        ("max_grad_spike", "Max Gradient Spike", "Worst-Case Instability"),
        ("loss_std", "Loss Std Dev", "Loss Stability (lower = more stable)"),
        ("margin_mean", "Mean Reward Margin", "Preference Discrimination"),
    ]
    
    for key, title, ylabel in plot_configs:
        fig, ax = plt.subplots(figsize=(8, 5))
        values = [m[key] for m in all_metrics]
        bar_colors = [colors.get(m, "#95a5a6") for m in methods]
        
        bars = ax.bar(methods, values, color=bar_colors, edgecolor="black", linewidth=0.5)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{val:.4f}", ha="center", va="bottom", fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"compare_{key}.png"), dpi=150)
        plt.close()

def generate_overlay_plots(all_logs, all_metrics):
    """Generate overlaid time-series plots with EMA smoothing for visual clarity."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    colors = {
        "DPO": "#e74c3c",          # Crimson Red
        "KTO": "#3498db",          # Blue
        # "VA-DPO (0.1)": "#85e3b2",  # Light Emerald
        # "VA-DPO (0.2)": "#2ecc71",  # Medium Emerald
        "VA-DPO (0.3)": "#1e8449",  # Dark Emerald
    }
    
    def compute_ema(values, alpha=0.25):
        if not values:
            return []
        smoothed = []
        curr = values[0]
        for val in values:
            curr = alpha * val + (1.0 - alpha) * curr
            smoothed.append(curr)
        return smoothed

    metric_configs = [
        ("grad_norm", "Gradient Norm Over Training"),
        ("loss", "Training Loss Over Training"),
        ("rewards/margins", "Reward Margins Over Training"),
    ]
    
    for metric_key, title in metric_configs:
        fig, ax = plt.subplots(figsize=(12, 5))
        
        for method_name, logs in all_logs.items():
            if method_name not in colors:
                continue
            steps = [e["step"] for e in logs if metric_key in e]
            values = [e[metric_key] for e in logs if metric_key in e]
            if not steps:
                continue
            color = colors[method_name]
            
            # Plot raw noisy background
            ax.plot(steps, values, color=color, alpha=0.2, linewidth=1.0, linestyle=":")
            
            # Plot smoothed foreground
            smoothed_values = compute_ema(values, alpha=0.3)
            ax.plot(steps, smoothed_values, label=method_name, color=color, linewidth=2.5)
        
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel(metric_key)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        safe_name = metric_key.replace("/", "_")
        plt.savefig(os.path.join(OUTPUT_DIR, f"overlay_{safe_name}.png"), dpi=150)
        plt.close()

def main():
    print("[START] Computing Stability Analysis")
    
    # Check which experiments exist
    available = {}
    for name, path in EXPERIMENTS.items():
        if os.path.exists(path):
            available[name] = path
        else:
            print(f"[SKIP] {name}: {path} not found")
    
    # Also check for VA-DPO results dynamically
    vadpo_dir = "outputs"
    if os.path.exists(vadpo_dir):
        for d in os.listdir(vadpo_dir):
            if d.startswith("vadpo_"):
                state_path = os.path.join(vadpo_dir, d, "checkpoint-200", "trainer_state.json")
                if os.path.exists(state_path):
                    lam = d.replace("vadpo_lambda_", "").replace("vadpo_", "")
                    available[f"VA-DPO ({lam})"] = state_path
    
    if len(available) < 2:
        print(f"[WARNING] Only {len(available)} experiment(s) found. Need at least 2 for comparison.")
        if len(available) == 0:
            return
    
    # Load all logs
    all_logs = {}
    all_metrics = []
    for name, path in available.items():
        print(f"Loading {name}...")
        logs = load_log_history(path)
        all_logs[name] = logs
        metrics = compute_stability_metrics(logs, name)
        all_metrics.append(metrics)
    
    # Print comparison table
    print_comparison_table(all_metrics)
    
    # Generate plots
    print("\nGenerating comparison plots...")
    generate_bar_plots(all_metrics)
    generate_overlay_plots(all_logs, all_metrics)
    
    # Save metrics JSON
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "stability_comparison.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\n[SUCCESS] All plots saved to {OUTPUT_DIR}/")
    print(f"[SUCCESS] Metrics saved to {OUTPUT_DIR}/stability_comparison.json")

if __name__ == "__main__":
    main()
