---
base_model: Qwen/Qwen2-1.5B-Instruct
library_name: peft
model_name: kto
tags:
- base_model:adapter:Qwen/Qwen2-1.5B-Instruct
- kto
- lora
- transformers
- trl
licence: license
pipeline_tag: text-generation
---

# Model Card for kto

This model is a fine-tuned version of [Qwen/Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 


This model was trained with KTO, a method introduced in [KTO: Model Alignment as Prospect Theoretic Optimization](https://huggingface.co/papers/2402.01306).

### Framework versions

- PEFT 0.18.1
- TRL: 0.23.0
- Transformers: 4.57.2
- Pytorch: 2.5.1+cu124
- Datasets: 4.8.4
- Tokenizers: 0.22.2

## Citations

Cite KTO as:

```bibtex
@article{ethayarajh2024kto,
    title        = {{KTO: Model Alignment as Prospect Theoretic Optimization}},
    author       = {Kawin Ethayarajh and Winnie Xu and Niklas Muennighoff and Dan Jurafsky and Douwe Kiela},
    year         = 2024,
    eprint       = {arXiv:2402.01306},
}
```

Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```