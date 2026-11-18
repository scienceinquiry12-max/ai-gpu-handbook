# AI GPU Handbook

**Practical answers to "which GPU, how much VRAM, and what will it cost" for running and training AI models.** It includes an LLM VRAM calculator (web and command line), an open GPU specs dataset, one-page cheat sheets and minimal code examples for local LLMs, RAG and QLoRA fine-tuning.

[![Web calculator](https://img.shields.io/badge/Try_it-LLM_VRAM_Calculator-4ed29c)](https://scienceinquiry12-max.github.io/ai-gpu-handbook/)
[![GPU prices](https://img.shields.io/badge/Live-GPU_cloud_prices-5b8cff)](https://tensorcore.si/gpu-cloud-pricing/)
[![License: MIT](https://img.shields.io/badge/code-MIT-lightgrey)](LICENSE)
[![Content: CC BY 4.0](https://img.shields.io/badge/data_%26_docs-CC_BY_4.0-lightgrey)](#license)

Maintained by [TensorCore.si](https://tensorcore.si/), which publishes plain-English guides to tensor cores, GPUs and AI hardware.

---

## Contents

- [Quick answer: how much VRAM do I need?](#quick-answer-how-much-vram-do-i-need)
- [LLM VRAM calculator](#llm-vram-calculator)
- [GPU specs dataset](#gpu-specs-dataset)
- [Cheat sheets](#cheat-sheets)
- [Code examples](#code-examples)
- [Learn the concepts](#learn-the-concepts)
- [Daily GPU cloud prices](#daily-gpu-cloud-prices)

## Quick answer: how much VRAM do I need?

Approximate memory to **run** a model for one user with an 8K context:

| Model size | FP16 | 8-bit | 4-bit (Q4_K_M) | Example GPU for 4-bit |
|---|---|---|---|---|
| 3B | ~7 GB | ~4 GB | ~3 GB | Any 6–8 GB GPU |
| 7–8B | ~17 GB | ~10 GB | ~6–7 GB | RTX 3060 12GB, RTX 5060 Ti 16GB |
| 13–14B | ~30 GB | ~16 GB | ~10 GB | RTX 5060 Ti 16GB, RTX 5070 Ti |
| 32B | ~68 GB | ~36 GB | ~22 GB | RTX 3090 / 4090 (24 GB), RTX 5090 |
| 70B | ~145 GB | ~76 GB | ~45–50 GB | 2× 24 GB, RTX PRO 6000, H100 80GB, 64 GB+ Mac |
| 405B | ~820 GB | ~420 GB | ~250 GB | 4× H100 (FP4) to 8× H100 / 4× B200 |

Longer contexts add KV cache on top: about **16 GB extra for Llama 3.1 8B at 128K tokens**. More in the [VRAM cheat sheet](cheatsheets/vram-cheatsheet.md).

## LLM VRAM calculator

**Web versions:** the full [VRAM calculator on TensorCore.si](https://tensorcore.si/vram-calculator/), or the lightweight [open-source version](https://scienceinquiry12-max.github.io/ai-gpu-handbook/) in this repo. Pick a model, precision, context length and task (run, QLoRA, LoRA or full fine-tune).

**More free tools on TensorCore.si:** [GPU specs database](https://tensorcore.si/gpu-specs/) · [Training cost calculator](https://tensorcore.si/training-cost-calculator/) · [LLM benchmarks](https://tensorcore.si/llm-benchmarks/) · [Number format converter](https://tensorcore.si/number-format-converter/) · [AI glossary](https://tensorcore.si/glossary/) · [All tools](https://tensorcore.si/tools/)

**Command line** (pure Python, no dependencies):

```bash
git clone https://github.com/scienceinquiry12-max/ai-gpu-handbook
cd ai-gpu-handbook
python tools/vram_calculator.py --model llama-3.1-8b
python tools/vram_calculator.py --model llama-3.1-70b --precision q4 --context 32768
python tools/vram_calculator.py --model qwen2.5-7b --mode qlora --context 2048 --batch 4
python tools/vram_calculator.py --params 13 --layers 40 --kv-heads 8 --precision fp8
python tools/vram_calculator.py --list-models
```

Example output:

```
llama-3.1-70b | inference | q4 | context 32,768 | batch 1

  weights                     39.9 GB
  kv cache                    10.0 GB
  overhead                     2.8 GB
  --------------------------------
  TOTAL (estimate)            52.7 GB

  Consumer / desktop that fit: NVIDIA DGX Spark (128 GB), Apple M4 Max (max config) (128 GB), ...
  Workstation / data center that fit: NVIDIA H100 SXM (80 GB), NVIDIA H100 PCIe (80 GB), ...
```

The methodology is explained in [How much VRAM do you need for an LLM?](https://tensorcore.si/how-much-vram-for-llm/)

## GPU specs dataset

[`data/gpu-specs.csv`](data/gpu-specs.csv) and [`data/gpu-specs.json`](data/gpu-specs.json) list 25 AI-relevant GPUs and accelerators, from the RTX 3060 to the B300 and MI355X, with memory, bandwidth, power, dense BF16/FP8 TFLOPS (data center parts) and launch prices (consumer parts).

| GPU | Memory | Bandwidth | Segment |
|---|---|---|---|
| NVIDIA B200 | 180 GB HBM3e | 8.0 TB/s | Data center |
| NVIDIA H200 SXM | 141 GB HBM3e | 4.8 TB/s | Data center |
| NVIDIA H100 SXM | 80 GB HBM3 | 3.35 TB/s | Data center |
| AMD MI300X | 192 GB HBM3 | 5.3 TB/s | Data center |
| RTX 5090 | 32 GB GDDR7 | 1.79 TB/s | Consumer |
| RTX 4090 | 24 GB GDDR6X | 1.01 TB/s | Consumer |
| DGX Spark | 128 GB unified | 273 GB/s | Desktop |

```python
import pandas as pd
gpus = pd.read_csv("https://raw.githubusercontent.com/scienceinquiry12-max/ai-gpu-handbook/main/data/gpu-specs.csv")
print(gpus.sort_values("bandwidth_gb_s", ascending=False).head())
```

Browse and compare more GPUs in the [GPU specs database](https://tensorcore.si/gpu-specs/). Found a wrong number or a missing GPU? [Open an issue](../../issues) or a pull request.

## Cheat sheets

| Cheat sheet | What it answers |
|---|---|
| [VRAM cheat sheet](cheatsheets/vram-cheatsheet.md) | Memory per model size, precision and context; what fits on 8–128 GB |
| [LLM math](cheatsheets/llm-math.md) | Training FLOPs (6ND), tokens/second from bandwidth, cost per million tokens |
| [Precision formats](cheatsheets/precision-formats.md) | FP32, TF32, BF16, FP16, FP8, INT8, MXFP4 and NVFP4 compared |
| [GPU buying guide for AI](cheatsheets/gpu-buying-guide.md) | Which GPU to buy for local LLMs, Stable Diffusion and fine-tuning |
| [AI and GPU glossary](cheatsheets/ai-glossary.md) | 50 terms (KV cache, MoE, RAG, HBM, NVLink...) in one line each |

## Code examples

| Example | What you'll build | Hardware |
|---|---|---|
| [Run an LLM locally](examples/run-llm-locally/) | Chat with Llama/Qwen through Ollama from Python and measure tokens/s | 8 GB+ GPU, Mac or CPU |
| [Minimal RAG](examples/rag-minimal/) | Embeddings, cosine-similarity retrieval and a grounded, cited answer | CPU is fine |
| [QLoRA fine-tuning](examples/qlora-finetune/) | Fine-tune an open LLM with TRL + PEFT in 4-bit | NVIDIA GPU, 8–24 GB |

## Learn the concepts

Beginner to advanced guides on [TensorCore.si](https://tensorcore.si/):

**AI basics:** [What is generative AI?](https://tensorcore.si/what-is-generative-ai/) · [What is an LLM?](https://tensorcore.si/what-is-an-llm/) · [How does ChatGPT work?](https://tensorcore.si/how-does-chatgpt-work/) · [What is a transformer?](https://tensorcore.si/what-is-a-transformer/) · [Tokens](https://tensorcore.si/what-are-tokens-in-ai/) · [Context window](https://tensorcore.si/what-is-a-context-window/) · [Embeddings](https://tensorcore.si/what-are-embeddings/) · [RAG](https://tensorcore.si/what-is-rag/) · [Fine-tuning](https://tensorcore.si/what-is-fine-tuning/) · [Mixture of experts](https://tensorcore.si/what-is-mixture-of-experts/) · [Reasoning models](https://tensorcore.si/what-is-a-reasoning-model/) · [AI hallucinations](https://tensorcore.si/what-is-ai-hallucination/)

**GPU hardware:** [What is a tensor core?](https://tensorcore.si/what-is-a-tensor-core/) · [Tensor cores vs CUDA cores](https://tensorcore.si/tensor-cores-vs-cuda-cores/) · [How a GPU works](https://tensorcore.si/how-a-gpu-works-sm-warps-memory/) · [What is VRAM?](https://tensorcore.si/what-is-vram/) · [HBM](https://tensorcore.si/what-is-hbm-high-bandwidth-memory/) · [Memory bandwidth](https://tensorcore.si/gpu-memory-bandwidth/) · [NVLink and NVSwitch](https://tensorcore.si/nvlink-nvswitch-explained/) · [Hopper vs Blackwell](https://tensorcore.si/hopper-vs-blackwell/) · [H100 vs H200](https://tensorcore.si/h100-vs-h200/) · [B200 vs H100](https://tensorcore.si/b200-vs-h100/)

**Performance:** [KV cache](https://tensorcore.si/kv-cache-explained/) · [Prefill vs decode](https://tensorcore.si/prefill-vs-decode/) · [FlashAttention](https://tensorcore.si/flashattention-explained/) · [Quantization: GGUF, GPTQ, AWQ](https://tensorcore.si/llm-quantization-explained-gguf-gptq-awq/) · [Training vs inference](https://tensorcore.si/ai-training-vs-inference/) · [Mixed precision training](https://tensorcore.si/mixed-precision-training/)

**Interactive simulators:** [Attention](https://tensorcore.si/attention-mechanism-simulator/) · [LLM inference](https://tensorcore.si/llm-inference-simulator/) · [Roofline model](https://tensorcore.si/roofline-model-simulator/) · [Sampling (temperature, top-p)](https://tensorcore.si/llm-sampling-simulator/)

**Buying and pricing:** [Best GPU for local LLMs](https://tensorcore.si/best-gpu-for-local-llm/) · [Best GPU for Stable Diffusion](https://tensorcore.si/best-gpu-for-stable-diffusion/) · [RTX 5090 for AI](https://tensorcore.si/rtx-5090-for-ai/) · [H100 price](https://tensorcore.si/h100-price/) · [Cheapest cloud GPU](https://tensorcore.si/cheapest-cloud-gpu/) · [Rent vs buy](https://tensorcore.si/rent-vs-buy-gpu/)

## Daily GPU cloud prices

On-demand prices for **H100, H200, B200 and A100** across 15+ cloud providers are collected every day and published at **[tensorcore.si/gpu-cloud-pricing](https://tensorcore.si/gpu-cloud-pricing/)**. The raw data and history are open in [tensorcore-gpu-prices](https://github.com/scienceinquiry12-max/tensorcore-gpu-prices).

## Contributing

Corrections, new GPUs, model presets and examples are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md). If this repo helped you, a ⭐ helps others find it.

## License

- **Code** (`tools/`, `examples/`, `docs/`): [MIT](LICENSE)
- **Data and cheat sheets** (`data/`, `cheatsheets/`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). You may reuse them with attribution to "AI GPU Handbook by TensorCore.si" and a link to <https://tensorcore.si/>.

Specifications come from manufacturer data sheets; estimates are approximate. Always verify before purchasing hardware.
