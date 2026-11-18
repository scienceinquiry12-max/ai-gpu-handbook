"""A minimal RAG (retrieval-augmented generation) pipeline in ~60 lines.

1. Embed a few documents with a small open embedding model.
2. Find the most similar passages to a question with cosine similarity.
3. Ask a local LLM (via Ollama) to answer using only those passages.

pip install sentence-transformers numpy requests
python rag.py "How much memory does an H200 have?"

Explainers: https://tensorcore.si/what-are-embeddings/ and https://tensorcore.si/what-is-rag/
"""
import sys

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

DOCS = [
    "The NVIDIA H100 SXM has 80 GB of HBM3 memory with 3.35 TB/s of bandwidth.",
    "The NVIDIA H200 has 141 GB of HBM3e memory with 4.8 TB/s of bandwidth and the same compute as the H100.",
    "The NVIDIA B200 (Blackwell) has about 180 GB of HBM3e per GPU in HGX systems and 8 TB/s of bandwidth.",
    "Tensor cores are units inside NVIDIA GPUs that perform matrix multiply-accumulate operations very quickly.",
    "The KV cache stores keys and values for previous tokens so they do not need to be recomputed.",
    "QLoRA fine-tunes small LoRA adapters on top of a model loaded in 4-bit precision.",
    "The RTX 5090 has 32 GB of GDDR7 memory and about 1.8 TB/s of memory bandwidth.",
]

question = sys.argv[1] if len(sys.argv) > 1 else "How much memory does an H200 have?"

# 1. Embed documents and question (normalized, so dot product = cosine similarity).
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
doc_vecs = embedder.encode(DOCS, normalize_embeddings=True)
q_vec = embedder.encode([question], normalize_embeddings=True)[0]

# 2. Retrieve the top 3 passages.
scores = doc_vecs @ q_vec
top = np.argsort(-scores)[:3]
context = "\n".join(f"[{i + 1}] {DOCS[j]}" for i, j in enumerate(top))
print("Retrieved:\n" + "\n".join(f"  {scores[j]:.3f}  {DOCS[j]}" for j in top) + "\n")

# 3. Generate an answer grounded in the retrieved passages.
prompt = (
    "Answer the question using only the sources below. "
    "Cite sources like [1]. If the answer is not in the sources, say you don't know.\n\n"
    f"Sources:\n{context}\n\nQuestion: {question}"
)
try:
    r = requests.post(
        "http://localhost:11434/api/chat",
        json={"model": "llama3.1:8b", "messages": [{"role": "user", "content": prompt}], "stream": False},
        timeout=600,
    )
    r.raise_for_status()
    print("Answer:\n" + r.json()["message"]["content"])
except requests.RequestException:
    print("Ollama is not running, so here is the prompt you would send to any LLM:\n\n" + prompt)
