import os

# Point to your existing cache to save space
os.environ["HF_HOME"] = r"S:\hf_cache"

import torch
import matplotlib.pyplot as plt
import pandas as pd
import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig

# 1. Config
MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"
OUTPUT_DIR = "./outputs/dpo_baseline"
SUBSET_SIZE = 5000  # As per research spec

def main():
    print("[START] Phase 2: DPO Baseline Training")
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Loading Dataset
    print(f"Dataset will be cached locally in ./datasets...")
    dataset = load_dataset(
        "trl-lib/ultrafeedback_binarized", 
        split="train",
        cache_dir="./datasets" # Local for intuitiveness
    )
    dataset = dataset.shuffle(seed=42).select(range(SUBSET_SIZE))

    def format_dpo(example):
        prompt_messages = example["chosen"][:-1]
        chosen_content = example["chosen"][-1]["content"]
        rejected_content = example["rejected"][-1]["content"]
        
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        # We include the assistant tags for the completions
        return {
            "prompt": prompt_text,
            "chosen": f"{chosen_content}<|im_end|>\n",
            "rejected": f"{rejected_content}<|im_end|>\n",
        }

    dataset = dataset.map(format_dpo, num_proc=4)

    # 3. Load Model (4-bit QLoRA)
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
    model.enable_input_require_grads() # Force gradients to flow for checkpointing

    # 4. LoRA Setup
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # 5. DPO Training Config
    # We use conservative settings for small-scale stability
    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=1e-6,
        beta=0.1,
        max_steps=200,             # Mini-run for verification; increase for full research
        logging_steps=10,
        save_steps=100,
        eval_strategy="no",
        optim="paged_adamw_32bit",
        remove_unused_columns=False,
        max_length=512,
        max_prompt_length=256,
        report_to="none",           # Set to "wandb" if you have it configured
        fp16=True,
    )

    # 6. Initialize Trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=None,             # TRL handles ref-model automatically for LoRA
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer, # Newer TRL versions use processing_class
        peft_config=peft_config,
    )

    print("--- Starting DPO Training ---")
    train_result = trainer.train()
    
    # 7. Plotting
    print("Generating research-grade training plots...")
    log_history = trainer.state.log_history
    df = pd.DataFrame(log_history)
    
    if not df.empty:
        os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)
        
        # Helper to clean up dataframes for plotting (remove Nones)
        def plot_metric(metric_name, title, color, filename):
            if metric_name in df.columns:
                sub_df = df[["step", metric_name]].dropna()
                if not sub_df.empty:
                    plt.figure(figsize=(10, 5))
                    plt.plot(sub_df["step"], sub_df[metric_name], label=title, color=color)
                    plt.title(f"DPO Baseline: {title}")
                    plt.xlabel("Step")
                    plt.ylabel(metric_name)
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    plt.savefig(os.path.join(OUTPUT_DIR, f"plots/{filename}"))
                    plt.close()

        # Generate the standard Research Suite
        plot_metric("loss", "Training Loss", "blue", "loss_curve.png")
        plot_metric("rewards/accuracies", "Reward Accuracy", "green", "accuracy_curve.png")
        plot_metric("grad_norm", "Gradient Norm (Fluctuations)", "red", "grad_norm.png")
        plot_metric("rewards/margins", "Reward Margin (Separation)", "purple", "reward_margin.png")
        
        # Calculate Stability Score (Standard Deviation of Loss)
        if "loss" in df.columns:
            loss_std = df["loss"].std()
            with open(os.path.join(OUTPUT_DIR, "stability_metrics.json"), "w") as f:
                json.dump({"loss_std_deviation": loss_std}, f)
            print(f"Stability Score (Loss Std): {loss_std:.4f}")
            
        print(f"Scientific plots saved to {OUTPUT_DIR}/plots/")

    # Save final model
    print(f"Saving baseline model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    print("[SUCCESS] Phase 2 Complete.")

if __name__ == "__main__":
    main()
