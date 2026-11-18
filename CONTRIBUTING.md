# Contributing

Thanks for helping make the AI GPU Handbook more accurate and useful.

- **Spec corrections:** open an issue or PR editing `data/gpu-specs.csv`, with a link to the manufacturer's data sheet. Then regenerate the JSON:
  `python -c "import csv,json;r=list(csv.DictReader(open('data/gpu-specs.csv')));json.dump(r,open('data/gpu-specs.json','w'),indent=1)"`
- **New model presets:** add the model to `MODELS` in `tools/vram_calculator.py` and `docs/index.html`, using the numbers from its `config.json` (layers, hidden size, KV heads, head dimension).
- **Examples:** keep them short, runnable and documented in a README.

Please keep the tone plain-English and beginner friendly.
