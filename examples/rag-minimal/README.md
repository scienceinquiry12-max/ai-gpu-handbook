# Minimal RAG with embeddings + a local LLM

The smallest useful retrieval-augmented generation example: embed documents, retrieve the closest ones to a question, and let an LLM answer only from them, with citations.

```bash
pip install sentence-transformers numpy requests
python rag.py "Which GPU has more memory, H100 or H200?"
```

- Runs on CPU. The embedding model is ~90 MB.
- For the answer step, start [Ollama](https://ollama.com) with `ollama run llama3.1:8b`. Without it, the script prints the grounded prompt instead.
- To scale up: chunk real documents, store vectors in pgvector, FAISS, Qdrant or similar, and add keyword (hybrid) search.

Learn more: [What are embeddings?](https://tensorcore.si/what-are-embeddings/) · [What is RAG?](https://tensorcore.si/what-is-rag/) · [Why AI hallucinates](https://tensorcore.si/what-is-ai-hallucination/)
