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
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import KTOTrainer, KTOConfig

# 1. Config (same as DPO baseline)
MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"
OUTPUT_DIR = "./outputs/kto"
SUBSET_SIZE = 5000  # Same subset as DPO for fair comparison

def main():
    print("[START] Phase 4: KTO Training")
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Loading Dataset
    # KTO needs UNPAIRED binary feedback: each row has (prompt, completion, label)
    # where label = True (desirable) or False (undesirable).
    # We convert the pairwise UltraFeedback into this format by splitting
    # each chosen/rejected pair into two separate rows.
    print("Dataset will be cached locally in ./datasets...")
    raw_dataset = load_dataset(
        "trl-lib/ultrafeedback_binarized", 
        split="train",
        cache_dir="./datasets"
    )
    raw_dataset = raw_dataset.shuffle(seed=42).select(range(SUBSET_SIZE))

    def convert_to_kto_format(dataset):
        """Convert pairwise (chosen/rejected) data into unpaired KTO format."""
        prompts = []
        completions = []
        labels = []
        
        for example in dataset:
            prompt_messages = example["chosen"][:-1]
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            
            chosen_content = example["chosen"][-1]["content"]
            rejected_content = example["rejected"][-1]["content"]
            
            # Add the CHOSEN as desirable (label=True)
            prompts.append(prompt_text)
            completions.append(f"{chosen_content}<|im_end|>\n")
            labels.append(True)
            
            # Add the REJECTED as undesirable (label=False)
            prompts.append(prompt_text)
            completions.append(f"{rejected_content}<|im_end|>\n")
            labels.append(False)
        
        return {
            "prompt": prompts,
            "completion": completions,
            "label": labels,
        }
    
    print("Converting to KTO format (unpaired binary feedback)...")
    kto_data = convert_to_kto_format(raw_dataset)
    
    from datasets import Dataset
    dataset = Dataset.from_dict(kto_data)
    dataset = dataset.shuffle(seed=42)
    print(f"KTO dataset size: {len(dataset)} rows ({SUBSET_SIZE} pairs -> {len(dataset)} unpaired samples)")

    # 3. Load Model (4-bit QLoRA) — identical to DPO baseline
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

    # 4. LoRA Setup — identical to DPO baseline
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # 5. KTO Training Config — same hyperparams as DPO baseline
    training_args = KTOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,  # KTO requires batch_size > 1 for KL estimation
        gradient_accumulation_steps=1,  # Effective batch = 2 (comparable to DPO's 1*2=2)
        learning_rate=1e-6,
        beta=0.1,                  # KL penalty weight (same as DPO)
        max_steps=200,             # Same as DPO for fair comparison
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

    # 6. Initialize KTO Trainer
    trainer = KTOTrainer(
        model=model,
        ref_model=None,             # TRL handles ref-model automatically for LoRA
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("--- Starting KTO Training ---")
    train_result = trainer.train()
    
    # 7. Plotting — same research suite as DPO
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
                    plt.title(f"KTO: {title}")
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
    print(f"Saving KTO model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    print("[SUCCESS] Phase 4 Complete.")

if __name__ == "__main__":
    main()
