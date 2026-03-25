"""Hybrid long-term memory module.

Provides the HybridMemory class: a persistent document store that combines
BM25 keyword search with ChromaDB semantic (vector) search, merged via
Reciprocal Rank Fusion.
"""

import json
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path


class HybridMemory:
    """Long-term memory store with hybrid BM25 and semantic retrieval.

    Combines ChromaDB cosine-similarity search with BM25 keyword scoring,
    merged via Reciprocal Rank Fusion (k=60, semantic weight 0.7).
    Falls back gracefully to BM25-only or keyword matching when
    optional dependencies are unavailable.
    """

    def __init__(self, persist_dir: str = "./memory_store"):
        self._lock = threading.Lock()
        self._docs: list[dict] = []
        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(exist_ok=True)
        self._load_existing()

        self._chroma = None
        self._embedder = None
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer

            client = chromadb.PersistentClient(path=str(self._persist_dir / "chroma"))
            self._collection = client.get_or_create_collection(
                name="agent_memory", metadata={"hnsw:space": "cosine"}
            )
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self._chroma = client
        except ImportError:
            print("[Memory] ChromaDB/sentence-transformers not found — BM25-only mode")

        self._bm25_corpus: list[list[str]] = []
        self._bm25_model = None
        self._rebuild_bm25()

    def _load_existing(self):
        """Restore previously persisted documents from disk."""
        db_path = self._persist_dir / "docs.json"
        if db_path.exists():
            with open(db_path) as f:
                self._docs = json.load(f)

    def _save_docs(self):
        """Persist the current document list to disk as JSON."""
        db_path = self._persist_dir / "docs.json"
        with open(db_path, "w") as f:
            json.dump(self._docs, f, indent=2)

    def _rebuild_bm25(self):
        """Rebuild the BM25 index from all stored documents."""
        if not self._docs:
            return
        try:
            import bm25s

            self._bm25_corpus = [doc["text"].lower().split() for doc in self._docs]
            self._bm25_model = bm25s.BM25()
            self._bm25_model.index(self._bm25_corpus)
        except ImportError:
            pass

    def store(self, text: str, metadata: dict) -> str:
        """Persist a document and update all search indices.

        Args:
            text: Document content to store.
            metadata: Arbitrary metadata attached to the document.

        Returns:
            Unique identifier for the stored document.
        """
        with self._lock:
            doc_id = str(uuid.uuid4())
            entry = {
                "id": doc_id,
                "text": text,
                "metadata": {
                    **metadata,
                    "timestamp": time.time(),
                    "datetime": datetime.now().isoformat(),
                },
            }
            self._docs.append(entry)
            self._save_docs()
            self._rebuild_bm25()

            if self._chroma and self._embedder:
                embedding = self._embedder.encode(text).tolist()
                self._collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[text],
                    metadatas=[entry["metadata"]],
                )
            return doc_id

    def search(
        self, query: str, top_k: int = 5, semantic_weight: float = 0.7
    ) -> list[dict]:
        """Retrieve documents ranked by Reciprocal Rank Fusion.

        Args:
            query: Natural-language search query.
            top_k: Maximum number of results to return.
            semantic_weight: Weight for vector search (BM25 gets 1 - weight).

        Returns:
            Sorted list of document dicts, most relevant first.
        """
        with self._lock:
            if not self._docs:
                return []

            results: dict[str, dict] = {}

            semantic_ids = []
            if self._chroma and self._embedder and len(self._docs) > 0:
                try:
                    query_emb = self._embedder.encode(query).tolist()
                    n = min(top_k * 2, len(self._docs))
                    sem_results = self._collection.query(
                        query_embeddings=[query_emb],
                        n_results=n,
                    )
                    semantic_ids = sem_results["ids"][0]
                    for rank, doc_id in enumerate(semantic_ids):
                        rrf_score = semantic_weight / (60 + rank + 1)
                        if doc_id not in results:
                            doc = next(
                                (d for d in self._docs if d["id"] == doc_id), None
                            )
                            if doc:
                                results[doc_id] = {"doc": doc, "score": 0.0}
                        if doc_id in results:
                            results[doc_id]["score"] += rrf_score
                except Exception:
                    pass

            if self._bm25_model and self._docs:
                try:
                    import bm25s

                    query_tokens = query.lower().split()
                    scores = self._bm25_model.get_scores(query_tokens)
                    ranked_indices = sorted(
                        range(len(scores)), key=lambda i: scores[i], reverse=True
                    )[: top_k * 2]
                    bm25_weight = 1 - semantic_weight
                    for rank, idx in enumerate(ranked_indices):
                        if idx < len(self._docs):
                            doc = self._docs[idx]
                            doc_id = doc["id"]
                            rrf_score = bm25_weight / (60 + rank + 1)
                            if doc_id not in results:
                                results[doc_id] = {"doc": doc, "score": 0.0}
                            results[doc_id]["score"] += rrf_score
                except Exception:
                    pass

            if not results and self._docs:
                query_lower = query.lower()
                for doc in self._docs:
                    if any(w in doc["text"].lower() for w in query_lower.split()):
                        results[doc["id"]] = {"doc": doc, "score": 0.1}

            sorted_results = sorted(
                results.values(), key=lambda x: x["score"], reverse=True
            )[:top_k]
            return [r["doc"] for r in sorted_results]

    def get_all(self, limit: int = 20) -> list[dict]:
        """Return the most recent documents up to *limit*."""
        with self._lock:
            return self._docs[-limit:]
