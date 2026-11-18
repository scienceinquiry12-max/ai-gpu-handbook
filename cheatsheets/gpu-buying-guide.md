# Which GPU should I buy for AI? (local LLMs, Stable Diffusion, fine-tuning)

For local AI, **VRAM decides what you can run; memory bandwidth decides how fast it runs.** Buy the most VRAM you can afford, and prefer NVIDIA unless you are comfortable with extra setup.

Prices are US launch prices; street prices in 2026 are often higher ([why GPU prices are rising](https://tensorcore.si/gpu-prices-rising-2026/)).

| Budget | Pick | VRAM | Good for |
|---|---|---|---|
| Entry | RTX 3060 12GB (used) | 12 GB | 7–8B LLMs, SD 1.5 / SDXL |
| Budget new | RTX 5060 Ti **16 GB** ($429) | 16 GB | 14B LLMs at Q4, FLUX (FP8/GGUF), QLoRA on 7B |
| Value | RTX 5070 Ti ($749) | 16 GB | Same models, much faster |
| Best VRAM per dollar | RTX 3090 (used) | 24 GB | 32B LLMs at Q4, FLUX, LoRA training |
| High end | RTX 5090 ($1,999) | 32 GB | 32B at Q6, full-precision FLUX, AI video |
| Big models at home | Mac with 64–512 GB unified memory, or DGX Spark ($3,999, 128 GB) | 64–512 GB | 70B+ and large MoE models (slower per token) |
| Workstation | RTX PRO 6000 Blackwell | 96 GB | 70B at Q8, serious fine-tuning |

## Avoid for AI

- **8 GB cards** (RTX 5060, 4060, 5060 Ti 8GB): too little memory for modern models.
- **Buying for TFLOPS alone:** a fast 12 GB card can't run what a slower 24 GB card can.

## NVIDIA vs AMD vs Apple

- **NVIDIA:** everything works out of the box with CUDA. → [What is CUDA?](https://tensorcore.si/what-is-cuda/)
- **AMD:** ROCm works, especially on Linux, but some tools and extensions don't. → [AMD vs NVIDIA for AI](https://tensorcore.si/amd-vs-nvidia-for-ai/)
- **Apple Silicon:** huge unified memory fits big models; generation is slower than high-end NVIDIA. Great with Ollama, LM Studio and MLX.

## Deeper guides

- [Best GPU for local LLMs](https://tensorcore.si/best-gpu-for-local-llm/)
- [Best GPU for Stable Diffusion and FLUX](https://tensorcore.si/best-gpu-for-stable-diffusion/)
- [Is the RTX 5090 good for AI?](https://tensorcore.si/rtx-5090-for-ai/)
- [How to read a GPU spec sheet](https://tensorcore.si/how-to-read-gpu-spec-sheet/)
- [Rent vs buy a GPU](https://tensorcore.si/rent-vs-buy-gpu/) and today's [cloud GPU prices](https://tensorcore.si/gpu-cloud-pricing/)
