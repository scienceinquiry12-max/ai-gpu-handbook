#!/usr/bin/env python3
"""
LLM VRAM calculator: estimate the GPU memory needed to run or fine-tune a
large language model, and see which GPUs it fits on.

No dependencies: standard-library Python 3.8+.

Examples:
    python tools/vram_calculator.py --model llama-3.1-8b
    python tools/vram_calculator.py --model llama-3.1-70b --precision q4 --context 32768
    python tools/vram_calculator.py --params 13 --precision fp16
    python tools/vram_calculator.py --model qwen2.5-7b --mode qlora --context 2048 --batch 4
    python tools/vram_calculator.py --list-models

Guide and background: https://tensorcore.si/how-much-vram-for-llm/
"""
import argparse
import csv
import math
import os
import sys

# Bytes per parameter for common weight formats.
# GGUF k-quants include block scales, so they use slightly more than the nominal bits.
PRECISION_BYTES = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "q8": 8.5 / 8,     # GGUF Q8_0
    "q6": 6.56 / 8,    # GGUF Q6_K
    "q5": 5.69 / 8,    # GGUF Q5_K_M
    "q4": 4.85 / 8,    # GGUF Q4_K_M
    "int4": 0.5,       # GPTQ / AWQ 4-bit (plus small overhead for scales)
    "fp4": 4.5 / 8,    # NVFP4 / MXFP4 including block scales
}

KV_BYTES = {"fp16": 2.0, "bf16": 2.0, "fp8": 1.0, "q8": 1.0, "q4": 0.5}

# params (billions), layers, hidden size, KV heads, head dimension.
# Architecture figures come from each model's published config.json.
MODELS = {
    "llama-3.1-8b":   dict(params=8.03,  layers=32,  hidden=4096,  kv_heads=8,  head_dim=128),
    "llama-3.1-70b":  dict(params=70.6,  layers=80,  hidden=8192,  kv_heads=8,  head_dim=128),
    "llama-3.1-405b": dict(params=405.9, layers=126, hidden=16384, kv_heads=8,  head_dim=128),
    "mistral-7b":     dict(params=7.24,  layers=32,  hidden=4096,  kv_heads=8,  head_dim=128),
    "qwen2.5-7b":     dict(params=7.62,  layers=28,  hidden=3584,  kv_heads=4,  head_dim=128),
    "qwen2.5-14b":    dict(params=14.8,  layers=48,  hidden=5120,  kv_heads=8,  head_dim=128),
    "qwen2.5-32b":    dict(params=32.8,  layers=64,  hidden=5120,  kv_heads=8,  head_dim=128),
    "qwen2.5-72b":    dict(params=72.7,  layers=80,  hidden=8192,  kv_heads=8,  head_dim=128),
    "phi-4-14b":      dict(params=14.7,  layers=40,  hidden=5120,  kv_heads=10, head_dim=128),
    "gemma-3-27b":    dict(params=27.4,  layers=62,  hidden=5376,  kv_heads=16, head_dim=128,
                           note="Gemma 3 uses sliding-window attention in most layers, so real KV cache is smaller."),
    "gpt-oss-20b":    dict(params=20.9,  layers=24,  hidden=2880,  kv_heads=8,  head_dim=64,
                           note="MoE: 3.6B active per token. Half the layers use a 128-token sliding window, so real KV cache is smaller."),
    "gpt-oss-120b":   dict(params=116.8, layers=36,  hidden=2880,  kv_heads=8,  head_dim=64,
                           note="MoE: 5.1B active per token. Half the layers use a 128-token sliding window, so real KV cache is smaller."),
}

GiB = 1024 ** 3


def kv_cache_bytes(layers, kv_heads, head_dim, context, batch, kv_bytes):
    """K and V vectors for every layer, KV head and token."""
    return 2 * layers * kv_heads * head_dim * kv_bytes * context * batch


def estimate(p, mode, precision, context, batch, kv_precision):
    params = p["params"] * 1e9
    out = {}
    if mode == "inference":
        out["weights"] = params * PRECISION_BYTES[precision]
        if p.get("layers"):
            out["kv_cache"] = kv_cache_bytes(p["layers"], p["kv_heads"], p["head_dim"],
                                             context, batch, KV_BYTES[kv_precision])
        else:
            out["kv_cache"] = 0.0
        # CUDA context, activations and framework buffers.
        out["overhead"] = 0.8 * GiB + 0.05 * out["weights"]
    else:
        hidden = p.get("hidden") or 4096
        layers = p.get("layers") or 32
        # Activations with gradient checkpointing (rough): layer inputs kept for
        # every layer plus one layer's full working set.
        act = batch * context * hidden * (layers * 2 + 34)
        if mode == "full":
            # Mixed precision + Adam: 2 (weights) + 2 (grads) + 4 (FP32 master) + 8 (Adam m, v)
            out["weights"] = params * 2
            out["gradients"] = params * 2
            out["optimizer"] = params * 12
        else:
            base = 2.0 if mode == "lora" else 0.52  # QLoRA: NF4 with double quantization
            out["weights"] = params * base
            lora_params = params * 0.01  # about 1% of parameters with r=16 on all linear layers
            out["adapters_and_optimizer"] = lora_params * 16
        out["activations"] = act
        out["overhead"] = 1.0 * GiB
    out["total"] = sum(out.values())
    return out


def load_gpus():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "gpu-specs.csv")
    try:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def fmt(b):
    return f"{b / GiB:8.1f} GB"


def main():
    ap = argparse.ArgumentParser(description="Estimate GPU memory for running or fine-tuning an LLM.")
    ap.add_argument("--model", help="preset name (see --list-models)")
    ap.add_argument("--params", type=float, help="parameters in billions, if not using a preset")
    ap.add_argument("--layers", type=int, help="number of transformer layers (for KV cache)")
    ap.add_argument("--kv-heads", type=int, help="number of key/value heads")
    ap.add_argument("--head-dim", type=int, default=128, help="attention head dimension (default 128)")
    ap.add_argument("--hidden", type=int, help="hidden size (for training activations)")
    ap.add_argument("--mode", default="inference", choices=["inference", "qlora", "lora", "full"])
    ap.add_argument("--precision", default="fp16", choices=sorted(PRECISION_BYTES),
                    help="weight format for inference (default fp16)")
    ap.add_argument("--kv-precision", default="fp16", choices=sorted(KV_BYTES))
    ap.add_argument("--context", type=int, default=8192, help="tokens of context (default 8192)")
    ap.add_argument("--batch", type=int, default=1, help="concurrent sequences / batch size (default 1)")
    ap.add_argument("--list-models", action="store_true")
    a = ap.parse_args()

    if a.list_models:
        for k, v in MODELS.items():
            print(f"{k:16s} {v['params']:7.1f}B  layers={v['layers']:<4d} kv_heads={v['kv_heads']}")
        return
    if a.model:
        key = a.model.lower()
        if key not in MODELS:
            sys.exit(f"Unknown model '{a.model}'. Use --list-models, or pass --params.")
        p = dict(MODELS[key])
    elif a.params:
        p = dict(params=a.params, layers=a.layers, hidden=a.hidden,
                 kv_heads=a.kv_heads or 8, head_dim=a.head_dim)
    else:
        ap.print_help()
        return

    r = estimate(p, a.mode, a.precision, a.context, a.batch, a.kv_precision)
    name = a.model or f"{a.params}B model"
    label = a.precision if a.mode == "inference" else a.mode.upper()
    print(f"\n{name} | {a.mode} | {label} | context {a.context:,} | batch {a.batch}\n")
    for k, v in r.items():
        if k != "total":
            print(f"  {k.replace('_', ' '):24s}{fmt(v)}")
    print(f"  {'-' * 32}\n  {'TOTAL (estimate)':24s}{fmt(r['total'])}\n")
    if p.get("note"):
        print(f"  Note: {p['note']}\n")
    if a.mode == "inference" and not p.get("layers"):
        print("  Tip: pass --layers and --kv-heads to include the KV cache.\n")

    gpus = load_gpus()
    if gpus:
        need = r["total"] / GiB
        groups = [("Consumer / desktop", ("consumer", "desktop", "laptop/desktop")),
                  ("Workstation / data center", ("workstation", "datacenter"))]
        for title, segs in groups:
            fits = [g for g in gpus if g["segment"] in segs and float(g["memory_gb"]) >= need]
            fits.sort(key=lambda g: float(g["memory_gb"]))
            names = ", ".join(f"{g['name']} ({g['memory_gb']} GB)" for g in fits[:6])
            print(f"  {title} that fit: {names or 'none'}")
        if need > 80:
            print("  Or split across several GPUs:")
            for n in ("NVIDIA H100 SXM", "NVIDIA H200 SXM", "NVIDIA B200"):
                g = next((x for x in gpus if x["name"] == n), None)
                if g:
                    # Splitting across GPUs adds overhead; assume 90% usable memory.
                    print(f"  {n}: about {math.ceil(need / (float(g['memory_gb']) * 0.9))} GPUs")
    print("\n  Estimates only. Real usage varies by framework, settings and drivers.")
    print("  Guide: https://tensorcore.si/how-much-vram-for-llm/ | Live GPU prices: https://tensorcore.si/gpu-cloud-pricing/\n")


if __name__ == "__main__":
    main()
