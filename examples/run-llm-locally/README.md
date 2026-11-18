# Run an LLM locally with Ollama (and call it from Python)

1. Install Ollama from <https://ollama.com/download>.
2. Pull and chat with a model:

```bash
ollama run llama3.1:8b        # ~4.9 GB download, needs ~6–8 GB VRAM or RAM
```

3. Call it from Python (Ollama serves an API on port 11434):

```bash
pip install requests
python chat.py "Explain tensor cores in two sentences."
```

Smaller GPU? Try `llama3.2:3b` or `qwen2.5:3b`. More VRAM? Try `qwen2.5:14b` or `gpt-oss:20b`.

Full step-by-step guide with LM Studio and llama.cpp: [How to run an LLM locally](https://tensorcore.si/run-llm-locally/) · How much memory you need: [VRAM cheat sheet](../../cheatsheets/vram-cheatsheet.md)
