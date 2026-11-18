"""Send a prompt to a local Ollama model and print the reply.

Usage: python chat.py "your question" [model]
Guide: https://tensorcore.si/run-llm-locally/
"""
import sys

import requests

prompt = sys.argv[1] if len(sys.argv) > 1 else "What is a tensor core?"
model = sys.argv[2] if len(sys.argv) > 2 else "llama3.1:8b"

r = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    },
    timeout=600,
)
r.raise_for_status()
data = r.json()
print(data["message"]["content"])

# Ollama reports timing in nanoseconds, so we can show generation speed.
if data.get("eval_count") and data.get("eval_duration"):
    print(f"\n[{data['eval_count'] / (data['eval_duration'] / 1e9):.1f} tokens/s]")
