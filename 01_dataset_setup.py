import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def load_and_prepare_dataset(tokenizer, subset_size=5000):
    """
    Loads UltraFeedback, formats it into prompt/chosen/rejected chat templates,
    and returns a filtered subset of a specific size. 
    """
    print(f"Loading ultrafeedback dataset (subset: {subset_size} samples)...")
    # Using trl-lib's pre-binarized version which is standard for DPO
    dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
    
    # Shuffle and select a subset
    dataset = dataset.shuffle(seed=42).select(range(subset_size))

    print(f"Dataset columns detected: {dataset.column_names}")

    def format_chat_template(example):
        # In trl-lib/ultrafeedback_binarized:
        # 'chosen' is a list of messages, where chosen[0] is the prompt and chosen[1:] is the response.
        # usually it's [system, user, assistant]
        
        # Split into prompt-part and response-part
        # Prompt: all messages except the last one
        prompt_messages = example["chosen"][:-1]
        
        # Response: the assistant's content
        chosen_content = example["chosen"][-1]["content"]
        rejected_content = example["rejected"][-1]["content"]
        
        # Apply Qwen2's chat template to the prompt
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        
        # Responses should just be the assistant tokens (including im_start/im_end if desired, 
        # but TRL often handles the wrap). For Qwen2, we want: <|im_start|>assistant\n{content}<|im_end|>\n
        # A clean way is to apply template to the whole thing and slice, OR format manually.
        
        # Let's format manually to be extremely precise for Qwen2
        def wrap_assistant(content):
            return f"{content}<|im_end|>\n" # DPOTrainer will add the <|im_start|>assistant tag if we don't, 
                                            # or we can include it. Qwen2-Instruct template uses:
                                            # <|im_start|>assistant\n{{content}}<|im_end|>\n
            
        return {
            "prompt": prompt_text,
            "chosen": wrap_assistant(chosen_content),
            "rejected": wrap_assistant(rejected_content)
        }

    print("Applying chat template formatting...")
    # Map the dataset
    formatted_dataset = dataset.map(format_chat_template, num_proc=4)
    
    # Filter out empty or broken pairs
    formatted_dataset = formatted_dataset.filter(
        lambda x: len(x["prompt"]) > 0 and len(x["chosen"]) > 0 and len(x["rejected"]) > 0
    )
    
    return formatted_dataset

def main():
    print("[START] Initializing Setup: Phase 1 (Model + Dataset)")
    model_id = "Qwen/Qwen2-1.5B-Instruct"

    print(f"Loading Tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Ensure correct padding token for Qwen2
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    dataset = load_and_prepare_dataset(tokenizer, subset_size=5000)
    print(f"\n[SUCCESS] Dataset loaded and formatted. Random sample size: {len(dataset)}")
    
    # Display a sample to confirm correctness
    sample = dataset[0]
    print("\n--- SAMPLE FORMAT ---")
    print(f"[PROMPT]\n{sample['prompt'][:200]}...\n")
    print(f"[CHOSEN]\n{sample['chosen'][:200]}...\n")
    print(f"[REJECTED]\n{sample['rejected'][:200]}...\n")
    print("---------------------\n")

    print(f"Verifying QLoRA configuration for {model_id}...")
    # Load model with quantization to verify the GPU + QLoRA setup works
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    print(f"Loading model with AutoModelForCausalLM...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    print(f"[SUCCESS] Model loaded successfully on device: {model.device}")
    
if __name__ == "__main__":
    main()
