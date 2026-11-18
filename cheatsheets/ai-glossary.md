# AI and GPU glossary: 50 terms explained in one line each

Plain-English definitions of the words you meet when working with LLMs and AI hardware. Each links to a longer explanation on [TensorCore.si](https://tensorcore.si/).

## Models and concepts

| Term | One-line definition |
|---|---|
| [Generative AI](https://tensorcore.si/what-is-generative-ai/) | AI that creates new text, images, audio, video or code. |
| [Large language model (LLM)](https://tensorcore.si/what-is-an-llm/) | A neural network trained on huge amounts of text to predict the next token. |
| [Neural network](https://tensorcore.si/what-is-a-neural-network/) | Layers of weighted connections that learn patterns from data. |
| [Transformer](https://tensorcore.si/what-is-a-transformer/) | The architecture behind modern LLMs, built around self-attention. |
| [Attention](https://tensorcore.si/attention-mechanism-simulator/) | The mechanism that lets each token draw information from other tokens. |
| [Token](https://tensorcore.si/what-are-tokens-in-ai/) | A chunk of text (≈ ¾ of an English word) that models read and write. |
| [Context window](https://tensorcore.si/what-is-a-context-window/) | The maximum number of tokens a model can consider at once. |
| [Embedding](https://tensorcore.si/what-are-embeddings/) | A vector of numbers that represents the meaning of data. |
| [RAG](https://tensorcore.si/what-is-rag/) | Retrieval-augmented generation: fetch relevant documents, then answer from them. |
| [Fine-tuning](https://tensorcore.si/what-is-fine-tuning/) | Further training a pre-trained model on your own examples. |
| LoRA / QLoRA | Fine-tuning small adapter matrices instead of all weights (QLoRA adds a 4-bit base). |
| RLHF | Reinforcement learning from human feedback, used to make models helpful. |
| [Mixture of experts (MoE)](https://tensorcore.si/what-is-mixture-of-experts/) | A model that routes each token to a few of many expert sub-networks. |
| [Reasoning model](https://tensorcore.si/what-is-a-reasoning-model/) | An LLM trained to think step by step before answering. |
| [AI agent](https://tensorcore.si/what-is-an-ai-agent/) | An LLM that plans and uses tools to complete multi-step tasks. |
| [Hallucination](https://tensorcore.si/what-is-ai-hallucination/) | A confident but false or invented output. |
| Temperature | A sampling setting: higher = more random, lower = more predictable. Try the [sampling simulator](https://tensorcore.si/llm-sampling-simulator/). |
| Parameters | The learned numbers (weights) inside a model; "8B" = 8 billion. |
| Pre-training | The expensive first training stage on trillions of tokens. |
| [Inference](https://tensorcore.si/ai-training-vs-inference/) | Running a trained model to produce outputs. |
| [AGI](https://tensorcore.si/what-is-agi/) | Hypothetical AI matching humans across most cognitive tasks. |
| [AI alignment](https://tensorcore.si/what-is-ai-alignment/) | Making AI systems pursue the goals their designers intend. |

## Inference and performance

| Term | One-line definition |
|---|---|
| [KV cache](https://tensorcore.si/kv-cache-explained/) | Stored keys and values of past tokens so they aren't recomputed. |
| [Prefill vs decode](https://tensorcore.si/prefill-vs-decode/) | Processing the prompt (compute-bound) vs generating tokens (memory-bound). |
| Time to first token (TTFT) | Delay before the first output token appears. |
| Tokens per second | Generation speed, mostly limited by memory bandwidth. |
| [Quantization](https://tensorcore.si/llm-quantization-explained-gguf-gptq-awq/) | Storing weights in fewer bits (8, 4) to save memory. |
| GGUF | The file format used by llama.cpp, Ollama and LM Studio. |
| [FlashAttention](https://tensorcore.si/flashattention-explained/) | A faster, memory-efficient way to compute attention. |
| Batching | Serving many requests together to keep the GPU busy. |
| [Roofline model](https://tensorcore.si/roofline-model-simulator/) | A chart showing whether a workload is compute- or memory-bound. |

## Hardware

| Term | One-line definition |
|---|---|
| [GPU](https://tensorcore.si/how-a-gpu-works-sm-warps-memory/) | A processor with thousands of cores for parallel math. |
| [CUDA](https://tensorcore.si/what-is-cuda/) | NVIDIA's platform for programming GPUs. |
| [CUDA cores](https://tensorcore.si/what-are-cuda-cores/) | General-purpose GPU arithmetic units. |
| [Tensor cores](https://tensorcore.si/what-is-a-tensor-core/) | Units that perform matrix multiply-accumulate very fast. |
| [VRAM](https://tensorcore.si/what-is-vram/) | Memory on the graphics card that holds model weights and data. |
| [HBM](https://tensorcore.si/what-is-hbm-high-bandwidth-memory/) | Stacked high-bandwidth memory used on data center GPUs. |
| [Memory bandwidth](https://tensorcore.si/gpu-memory-bandwidth/) | How fast data moves between memory and compute (TB/s). |
| [TFLOPS vs TOPS](https://tensorcore.si/tops-vs-tflops/) | Trillions of floating-point vs integer operations per second. |
| [NVLink / NVSwitch](https://tensorcore.si/nvlink-nvswitch-explained/) | NVIDIA's high-speed GPU-to-GPU interconnect. |
| [GPU cluster](https://tensorcore.si/what-is-a-gpu-cluster/) | Many GPU servers networked together to train or serve models. |
| [AI data center](https://tensorcore.si/what-is-an-ai-data-center/) | A facility built for dense, power-hungry AI servers. |
| [TPU](https://tensorcore.si/tpu-vs-gpu/) | Google's custom AI accelerator. |
| [NPU](https://tensorcore.si/npu-vs-gpu-vs-tpu/) | A low-power AI engine in phones and laptops. |
| [AI PC](https://tensorcore.si/what-is-an-ai-pc/) | A computer with an NPU for on-device AI features. |
| [Hopper vs Blackwell](https://tensorcore.si/hopper-vs-blackwell/) | NVIDIA's 2022 (H100/H200) and 2024–25 (B200/B300) data center architectures. |
| [GB200 NVL72](https://tensorcore.si/gb200-nvl72-explained/) | A rack of 72 Blackwell GPUs linked as one NVLink domain. |
| [Precision formats](https://tensorcore.si/fp16-bf16-fp8-fp4-precision-formats/) | FP32, BF16, FP16, FP8, FP4: how many bits each number uses. |
| [Mixed precision](https://tensorcore.si/mixed-precision-training/) | Training in BF16/FP8 while keeping an FP32 master copy. |
| [Structured sparsity](https://tensorcore.si/structured-sparsity-explained/) | 2:4 pruning that tensor cores can skip for up to 2× speed. |
