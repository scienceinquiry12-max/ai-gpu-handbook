# LLM math cheat sheet: memory, speed, FLOPs and cost

Back-of-the-envelope formulas used by ML engineers to size hardware. N = parameters, D = training tokens.

## Memory

| What | Formula |
|---|---|
| Weights | N × bytes per parameter |
| KV cache | 2 × layers × KV heads × head dim × bytes × tokens × batch |
| Full training (mixed precision + Adam) | ≈ 16 bytes × N + activations |
| QLoRA fine-tuning | ≈ 0.5 bytes × N + small adapters + activations |

→ [How much VRAM for an LLM](https://tensorcore.si/how-much-vram-for-llm/) · [KV cache explained](https://tensorcore.si/kv-cache-explained/)

## Compute

| What | Formula |
|---|---|
| Training compute | ≈ 6 × N × D FLOPs |
| Inference compute | ≈ 2 × N FLOPs per token |
| Compute-optimal data (Chinchilla) | D ≈ 20 × N tokens |
| Training time | 6ND ÷ (GPUs × peak FLOPS × utilization) |

Example: an 8B model trained on 15T tokens needs about 6 × 8e9 × 15e12 = 7.2 × 10²³ FLOPs. At 40% utilization of 989 dense BF16 TFLOPS per H100, that is about 0.5 million H100-hours (roughly 21 days on 1,000 GPUs).

→ [AI training vs inference](https://tensorcore.si/ai-training-vs-inference/) · [LLM training cost](https://tensorcore.si/llm-training-cost/)

## Speed (single user)

Token generation (decode) is usually memory-bandwidth-bound:

```
max tokens/second ≈ memory bandwidth (GB/s) ÷ bytes read per token (≈ model size in GB)
```

| GPU | Bandwidth | 8B Q4 (~4.8 GB) | 70B Q4 (~42 GB) |
|---|---|---|---|
| RTX 4090 | 1,008 GB/s | ~210 tok/s upper bound | doesn't fit |
| RTX 5090 | 1,792 GB/s | ~370 tok/s upper bound | doesn't fit |
| H100 SXM | 3,350 GB/s | ~700 tok/s upper bound | ~80 tok/s upper bound |
| Apple M4 Max | 546 GB/s | ~110 tok/s upper bound | ~13 tok/s upper bound |

Real speeds are typically 50–80% of these upper bounds. For MoE models, use the size of the *active* parameters.

→ [GPU memory bandwidth explained](https://tensorcore.si/gpu-memory-bandwidth/) · [Prefill vs decode](https://tensorcore.si/prefill-vs-decode/) · [Roofline model simulator](https://tensorcore.si/roofline-model-simulator/)

## Cost

```
cost per 1M output tokens ≈ GPU price per hour ÷ (tokens per second × 3,600) × 1,000,000
```

Example: one H100 at $2.50/hour serving 1,500 tokens/second in aggregate (batched) ≈ $0.46 per million tokens.

Live H100, H200, B200 and A100 rental prices: [GPU cloud price tracker](https://tensorcore.si/gpu-cloud-pricing/)
