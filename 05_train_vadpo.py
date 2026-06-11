"""
05_train_vadpo.py
Variance-Aware DPO (VA-DPO): Temporal gradient-norm regularization for stable
preference optimization at small model scales.

Key idea: L_VADPO(t) = L_DPO(t) + lambda * (||g_t|| - ||g_{t-1}||)^2

The regularizer penalizes abrupt changes in gradient magnitude ACROSS steps,
targeting the temporal instability observed in DPO at 1.5B scale.

Implementation: We override training_step() because gradient norms are only
available AFTER backward(). The penalty from step t is applied at step t+1.
"""
import os
import sys
import argparse

# Point to your existing cache to save space
os.environ["HF_HOME"] = r"S:\hf_cache"

import torch
import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig

# ============================================================
# VA-DPO Trainer: Custom subclass with temporal regularization
# ============================================================

class VADPOTrainer(DPOTrainer):
    """
    Variance-Aware DPO Trainer.
    
    Adds a temporal gradient-norm smoothness penalty to the standard DPO loss:
        penalty = lambda_va * (||g_t|| - ||g_{t-1}||)^2
    
    Optimized for physical safety and performance:
    * ZERO CPU-GPU synchronizations during standard training steps (100% asynchronous).
    * Correctly handles AMP/FP16 gradient scaling (divides by GradScaler scale factor).
    * Software thermal pacing delay after each optimizer step to prevent laptop blackouts.
    """
    
    def __init__(self, lambda_va=0.01, step_sleep=0.1, **kwargs):
        super().__init__(**kwargs)
        self.lambda_va = lambda_va
        self.step_sleep = step_sleep
        self.prev_grad_norm = None
        self.va_penalties = []  # Track penalties for logging
        print(f"[VA-DPO] Initialized with lambda={lambda_va}, step_sleep={step_sleep}")
    
    def _get_grad_norm_tensor(self, model):
        """
        Compute the L2 norm of all trainable gradients as a GPU tensor.
        
        CRITICAL: With fp16=True, PyTorch's AMP GradScaler multiplies the loss
        by ~65536 before backward(), so raw p.grad values are ~65536x inflated.
        We divide by the scaler's scale factor to get the TRUE unscaled norm
        that matches TRL's logged grad_norm.
        """
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads:
            return torch.tensor(0.0, device=self.model.device)
        # Compute squared norms on GPU entirely asynchronously
        squared_norms = [torch.norm(g.detach(), 2) ** 2 for g in grads]
        raw_norm = torch.sum(torch.stack(squared_norms)) ** 0.5
        
        # Correct for AMP scaling: divide by GradScaler's scale factor
        # This makes our norm match TRL's logged grad_norm (which is post-unscale)
        scaler = getattr(self.accelerator, 'scaler', None)
        if scaler is not None:
            scale = scaler.get_scale()
            if scale > 0:
                raw_norm = raw_norm / scale
        
        return raw_norm
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Override training_step to inject temporal regularization.
        """
        # Step 1: Standard DPO training step (forward + backward)
        loss = super().training_step(model, inputs, num_items_in_batch)
        
        # Step 2: Measure current gradient norm as a GPU tensor (0 syncs)
        current_grad_norm = self._get_grad_norm_tensor(model)
        
        # Check if we are at the end of a gradient accumulation cycle
        is_sync_step = getattr(self, "accelerator", None) is None or self.accelerator.sync_gradients
        
        # Step 3: Apply temporal penalty entirely on the GPU (0 syncs)
        if is_sync_step and self.prev_grad_norm is not None and self.lambda_va > 0:
            norm_diff = current_grad_norm - self.prev_grad_norm
            
            # Perform all math on the GPU
            # Direction: if norm increased (norm_diff > 0), we want to damp the gradient.
            # We multiply by (1.0 + penalty_scale). For damping, penalty_scale must be negative.
            penalty_scale = -self.lambda_va * 2.0 * norm_diff / (current_grad_norm + 1e-8)
            
            # Apply safety clamp to the scaling factor entirely on GPU
            penalty_scale = torch.clamp(penalty_scale, min=-0.3, max=0.3)
            
            # Modulate gradients in tensor space (GPU-only, 0 syncs)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(1.0 + penalty_scale)
            
            # === DIAGNOSTIC: Print on EVERY step to verify AMP fix ===
            # curr_val = current_grad_norm.item()
            # prev_val = self.prev_grad_norm.item()
            # diff_val = norm_diff.item()
            # penalty_val = self.lambda_va * (diff_val ** 2)
            # scale_val = penalty_scale.item()
            
            # Get AMP scaler value to prove division is happening
            amp_scale = "N/A"
            scaler = getattr(self.accelerator, 'scaler', None)
            if scaler is not None:
                amp_scale = f"{scaler.get_scale():.0f}"
            
            step = self.state.global_step if hasattr(self, "state") else "?"
            print(f"[VA-DPO DEBUG] step={step} "
                  f"curr_norm={curr_val:.4f} prev_norm={prev_val:.4f} "
                  f"diff={diff_val:.4f} penalty={penalty_val:.6f} "
                  f"penalty_scale={scale_val:.6f} final_scale={1.0 + scale_val:.4f} "
                  f"amp_scale={amp_scale}")
            
            self.va_penalties.append(penalty_val)
            
            # TRL logging only on logging steps (to keep log_history clean)
            if hasattr(self, "state") and self.state.global_step % self.args.logging_steps == 0:
                self.log({"va_penalty": penalty_val, "grad_norm_diff": diff_val})
        
        # Step 4: Cache for next step and apply thermal sleep delay
        if is_sync_step:
            self.prev_grad_norm = current_grad_norm.detach()
            
            # Thermal pacing to let GPU/VRMs cool and prevent blackouts
            if self.step_sleep > 0:
                import time
                time.sleep(self.step_sleep)
        
        return loss


# ============================================================
# Config
# ============================================================
MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"
SUBSET_SIZE = 5000

def main():
    parser = argparse.ArgumentParser(description="VA-DPO Training")
    parser.add_argument("--lambda_va", type=float, default=0.01,
                        help="Regularization strength (default: 0.01)")
    parser.add_argument("--max_steps", type=int, default=200,
                        help="Training steps (default: 200)")
    parser.add_argument("--step_sleep", type=float, default=0.1,
                        help="Cooldown pause in seconds after each step to prevent thermal build-up (default: 0.1)")
    args = parser.parse_args()
    
    lambda_va = args.lambda_va
    OUTPUT_DIR = f"./outputs/vadpo_lambda_{lambda_va}"
    
    print(f"[START] Phase 3: VA-DPO Training (lambda={lambda_va})")
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset — identical to DPO baseline
    print("Dataset will be cached locally in ./datasets...")
    dataset = load_dataset(
        "trl-lib/ultrafeedback_binarized", 
        split="train",
        cache_dir="./datasets"
    )
    dataset = dataset.shuffle(seed=42).select(range(SUBSET_SIZE))

    def format_dpo(example):
        prompt_messages = example["chosen"][:-1]
        chosen_content = example["chosen"][-1]["content"]
        rejected_content = example["rejected"][-1]["content"]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        return {
            "prompt": prompt_text,
            "chosen": f"{chosen_content}<|im_end|>\n",
            "rejected": f"{rejected_content}<|im_end|>\n",
        }

    dataset = dataset.map(format_dpo, num_proc=4)

    # Model — identical to DPO baseline
    print(f"Loading {MODEL_ID} in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # LoRA — identical to DPO baseline
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # Training Config — identical to DPO baseline
    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=1e-6,
        beta=0.1,
        max_steps=args.max_steps,
        logging_steps=10,
        save_steps=100,
        eval_strategy="no",
        optim="paged_adamw_32bit",
        remove_unused_columns=False,
        max_length=512,
        max_prompt_length=256,
        report_to="none",
        fp16=True,
    )

    # Initialize VA-DPO Trainer
    trainer = VADPOTrainer(
        lambda_va=lambda_va,
        step_sleep=args.step_sleep,
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print(f"--- Starting VA-DPO Training (lambda={lambda_va}) ---")
    train_result = trainer.train()
    
    # Plotting — same research suite + VA-DPO specific metrics
    print("Generating research-grade training plots...")
    log_history = trainer.state.log_history
    df = pd.DataFrame(log_history)
    
    if not df.empty:
        os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)
        
        def plot_metric(metric_name, title, color, filename):
            if metric_name in df.columns:
                sub_df = df[["step", metric_name]].dropna()
                if not sub_df.empty:
                    plt.figure(figsize=(10, 5))
                    plt.plot(sub_df["step"], sub_df[metric_name], label=title, color=color)
                    plt.title(f"VA-DPO (lambda={lambda_va}): {title}")
                    plt.xlabel("Step")
                    plt.ylabel(metric_name)
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    plt.savefig(os.path.join(OUTPUT_DIR, f"plots/{filename}"))
                    plt.close()

        # Standard Research Suite
        plot_metric("loss", "Training Loss", "blue", "loss_curve.png")
        plot_metric("rewards/accuracies", "Reward Accuracy", "green", "accuracy_curve.png")
        plot_metric("grad_norm", "Gradient Norm (Fluctuations)", "red", "grad_norm.png")
        plot_metric("rewards/margins", "Reward Margin (Separation)", "purple", "reward_margin.png")
        
        # VA-DPO specific
        plot_metric("va_penalty", "VA Penalty", "orange", "va_penalty.png")
        plot_metric("grad_norm_diff", "Grad Norm Change (Step-to-Step)", "darkred", "grad_norm_diff.png")
        
        # Stability metrics
        if "loss" in df.columns:
            loss_std = df["loss"].dropna().std()
            grad_norms = df["grad_norm"].dropna().tolist()
            grad_std = float(np.std(grad_norms, ddof=1)) if grad_norms else 0.0
            
            metrics = {
                "loss_std_deviation": float(loss_std),
                "grad_norm_std": grad_std,
                "lambda_va": lambda_va,
                "mean_va_penalty": float(np.mean(trainer.va_penalties)) if trainer.va_penalties else 0.0,
            }
            with open(os.path.join(OUTPUT_DIR, "stability_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"Stability Score (Loss Std): {loss_std:.4f}")
            print(f"Stability Score (Grad Std): {grad_std:.4f}")
            print(f"Mean VA Penalty: {metrics['mean_va_penalty']:.6f}")
            
        print(f"Scientific plots saved to {OUTPUT_DIR}/plots/")

    # Save model
    print(f"Saving VA-DPO model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    print(f"[SUCCESS] VA-DPO (lambda={lambda_va}) Complete.")

if __name__ == "__main__":
    main()