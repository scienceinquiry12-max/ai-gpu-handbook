# LLM VRAM cheat sheet

How much GPU memory do you need to run a large language model? Start here, then use the [VRAM calculator](../tools/vram_calculator.py) for your exact model and context length.

> Full guide: [How much VRAM do you need for an LLM?](https://tensorcore.si/how-much-vram-for-llm/) on TensorCore.si

## Rule of thumb

```
Weights (GB) ≈ parameters (billions) × bytes per parameter
Total       ≈ weights + KV cache + 1–2 GB overhead
```

| Format | Bytes per parameter | 8B model | 70B model |
|---|---|---|---|
| FP32 | 4 | 32 GB | 280 GB |
| FP16 / BF16 | 2 | 16 GB | 140 GB |
| FP8 / INT8 | 1 | 8 GB | 70 GB |
| GGUF Q8_0 | ~1.06 | 8.5 GB | 74 GB |
| GGUF Q4_K_M | ~0.61 | 4.8 GB | 42 GB |
| NVFP4 / MXFP4 | ~0.56 | 4.5 GB | 39 GB |

## Weights only, by model size (GB, decimal)

| Model size | FP16 | FP8 | Q8_0 | Q4_K_M | FP4 |
|---|---|---|---|---|---|
| 1B | 2.0 | 1.0 | 1.1 | 0.6 | 0.6 |
| 3B | 6.0 | 3.0 | 3.2 | 1.8 | 1.7 |
| 7–8B | 16.0 | 8.0 | 8.5 | 4.8 | 4.5 |
| 13–14B | 28.0 | 14.0 | 14.9 | 8.5 | 7.9 |
| 32B | 64.0 | 32.0 | 34.0 | 19.4 | 18.0 |
| 70B | 140.0 | 70.0 | 74.4 | 42.4 | 39.4 |
| 120B | 240.0 | 120.0 | 127.5 | 72.8 | 67.5 |
| 405B | 810.0 | 405.0 | 430.3 | 245.5 | 227.8 |

Add the KV cache and about 1–2 GB of overhead on top.

## KV cache (the part people forget)

```
KV cache (bytes) = 2 × layers × KV heads × head dim × bytes per value × tokens × batch
```

| Model | Per token (FP16) | 8K context | 32K context | 128K context |
|---|---|---|---|---|
| Llama 3.1 8B | 128 KB | 1 GB | 4 GB | 16 GB |
| Llama 3.1 70B | 320 KB | 2.5 GB | 10 GB | 40 GB |
| Qwen2.5 7B | 56 KB | 0.44 GB | 1.75 GB | 7 GB |

Halve these numbers with an FP8 / Q8 KV cache. Explainer: [KV cache explained](https://tensorcore.si/kv-cache-explained/).

## What fits on which GPU (single user, 8K context)

| GPU memory | Comfortable models |
|---|---|
| 8 GB | 7–8B at Q4 |
| 12 GB | 7–8B at Q8, 13–14B at Q4 |
| 16 GB | 13–14B at Q4–Q6, 20B MoE (gpt-oss-20b) |
| 24 GB | 32B at Q4, 8B at FP16 |
| 32 GB | 32B at Q5–Q6, 70B only with heavy quantization (Q2–Q3) |
| 48 GB | 70B at Q4 |
| 80 GB | 70B at Q6–Q8, gpt-oss-120b (MXFP4) |
| 128 GB unified (Mac / DGX Spark) | 70B at Q8, 120B MoE at Q4–FP4 |

## Fine-tuning memory (rough)

| Method | Bytes per parameter | 7–8B model |
|---|---|---|
| Full fine-tuning (mixed precision + Adam) | ~16 + activations | 120 GB+ |
| LoRA (16-bit base) | ~2 + activations | 18–24 GB |
| QLoRA (4-bit base) | ~0.5 + activations | 8–12 GB |

Guide: [What is fine-tuning? LoRA and QLoRA explained](https://tensorcore.si/what-is-fine-tuning/)

---

Need a GPU for a few hours instead? Compare daily on-demand prices for H100, H200, B200 and A100 in the [GPU cloud price tracker](https://tensorcore.si/gpu-cloud-pricing/).
