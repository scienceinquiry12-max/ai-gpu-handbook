# AI number formats cheat sheet: FP32, TF32, BF16, FP16, FP8, FP4, INT8

Lower precision means smaller models, less memory traffic and faster tensor-core math, at some cost in accuracy.

> Full guide: [FP16 vs BF16 vs FP8 vs FP4 precision formats](https://tensorcore.si/fp16-bf16-fp8-fp4-precision-formats/)

| Format | Bits | Exponent / mantissa | Range | Main use | First NVIDIA tensor-core support |
|---|---|---|---|---|---|
| FP32 | 32 | 8 / 23 | Huge | Master weights, optimizer states | (CUDA cores) |
| TF32 | 19 (stored as 32) | 8 / 10 | Same as FP32 | Drop-in FP32 training speed-up | Ampere (A100) |
| BF16 | 16 | 8 / 7 | Same as FP32 | Default for training LLMs | Ampere (A100) |
| FP16 | 16 | 5 / 10 | ±65,504 | Inference, older training (needs loss scaling) | Volta (V100) |
| FP8 E4M3 | 8 | 4 / 3 | ±448 | Inference, forward pass in training | Hopper (H100) |
| FP8 E5M2 | 8 | 5 / 2 | ±57,344 | Gradients in FP8 training | Hopper (H100) |
| INT8 | 8 | integer | −128…127 | Quantized inference | Turing (T4) |
| MXFP4 | 4 + shared scale | 2 / 1 | Block-scaled | Open-weight models (gpt-oss) | Blackwell |
| NVFP4 | 4 + FP8 scale per 16 values | 2 / 1 | Block-scaled | Blackwell inference | Blackwell (B200, RTX 50) |

## Quick rules

- **Training:** BF16 with FP32 master weights is the safe default. FP8 training works on Hopper/Blackwell with libraries like Transformer Engine. → [Mixed precision training](https://tensorcore.si/mixed-precision-training/)
- **Serving on data center GPUs:** FP8 is close to lossless for most LLMs; NVFP4 roughly halves memory again on Blackwell. → [NVFP4 explained](https://tensorcore.si/nvfp4-explained/)
- **Running locally:** GGUF Q4_K_M is the usual sweet spot; Q5/Q6 if you have room; Q8 is near-lossless. → [LLM quantization: GGUF, GPTQ, AWQ](https://tensorcore.si/llm-quantization-explained-gguf-gptq-awq/)
- **FP16 overflow:** values above 65,504 become infinity. BF16 avoids this, which is why it replaced FP16 for training.

## Memory per billion parameters

| Format | GB per 1B params |
|---|---|
| FP32 | 4.0 |
| BF16 / FP16 | 2.0 |
| FP8 / INT8 | 1.0 |
| 4-bit (INT4, NF4) | ~0.5 |
| FP4 with scales | ~0.56 |

Tensor cores explained: [What is a tensor core?](https://tensorcore.si/what-is-a-tensor-core/)
