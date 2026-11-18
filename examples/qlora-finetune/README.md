# QLoRA fine-tuning on one GPU

Fine-tune an open LLM by training small LoRA adapters on top of a 4-bit base model, using Hugging Face TRL and PEFT.

```bash
pip install -r requirements.txt
python train.py
```

| Base model | Approx. VRAM (batch 2, 1K tokens) |
|---|---|
| Qwen2.5-0.5B (default) | ~4–6 GB |
| Qwen2.5-7B / Llama 3.1 8B | ~12–16 GB |
| Qwen2.5-14B | ~18–24 GB |

Needs an NVIDIA GPU with BF16 support (RTX 30-series or newer). Estimate other sizes with the [VRAM calculator](../../tools/vram_calculator.py): `python tools/vram_calculator.py --model qwen2.5-7b --mode qlora --context 1024 --batch 2`

No suitable GPU? Rent one by the hour; compare prices in the [GPU cloud price tracker](https://tensorcore.si/gpu-cloud-pricing/).

Learn more: [What is fine-tuning? LoRA, QLoRA and when to use RAG instead](https://tensorcore.si/what-is-fine-tuning/)
