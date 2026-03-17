"""
vector_database/vector_store.py
=================================
Unified interface for Chroma, FAISS, and Qdrant vector stores.

Each backend implements BaseVectorStore:
    index(chunks, embeddings)          → None
    query(embedding, top_k)            → list[str]
    get_chunk_count()                  → int
    clear()                            → None

Usage
-----
    from vector_database.vector_store import get_vector_store
    store = get_vector_store("faiss", dimension=384)
    store.index(chunks, vectors)
    results = store.query(query_vec, top_k=5)
"""

from __future__ import annotations

import abc
import uuid
from typing import Optional

import numpy as np
from loguru import logger


# ── Base Interface ─────────────────────────────────────────────────────────────

class BaseVectorStore(abc.ABC):
    """Abstract base class for all vector store backends."""

    name: str = "base"

    @abc.abstractmethod
    def index(self, chunks: list[str], embeddings: np.ndarray) -> None:
        """Store *chunks* with their *embeddings*."""

    @abc.abstractmethod
    def query(self, embedding: np.ndarray, top_k: int = 5) -> list[str]:
        """Return the top-k most similar chunks for an *embedding* query."""

    @abc.abstractmethod
    def get_chunk_count(self) -> int:
        """Return the total number of indexed chunks."""

    @abc.abstractmethod
    def clear(self) -> None:
        """Remove all indexed data."""


# ── FAISS ──────────────────────────────────────────────────────────────────────

class FAISSVectorStore(BaseVectorStore):
    """In-memory FAISS flat L2 / cosine index."""

    name = "faiss"

    def __init__(self, dimension: Optional[int] = None):
        self.dimension = dimension
        self._index = None
        self._chunks: list[str] = []

    def _init_index(self, dim: int):
        try:
            import faiss
            self._index = faiss.IndexFlatIP(dim)  # inner product = cosine if normalised
            self.dimension = dim
        except ImportError:
            logger.error("faiss-cpu not installed. Run: pip install faiss-cpu")
            raise

    def index(self, chunks: list[str], embeddings: np.ndarray) -> None:
        dim = embeddings.shape[1]
        if self._index is None or self.dimension != dim:
            self._init_index(dim)
            self._chunks = []

        import faiss
        vecs = np.asarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(vecs)
        self._index.add(vecs)
        self._chunks.extend(chunks)
        logger.debug(f"FAISS: indexed {len(chunks)} chunks (total={self._index.ntotal})")

    def query(self, embedding: np.ndarray, top_k: int = 5) -> list[str]:
        if self._index is None or self._index.ntotal == 0:
            return []
        import faiss
        vec = np.asarray([embedding], dtype=np.float32)
        faiss.normalize_L2(vec)
        k = min(top_k, self._index.ntotal)
        _, indices = self._index.search(vec, k)
        return [self._chunks[i] for i in indices[0] if 0 <= i < len(self._chunks)]

    def get_chunk_count(self) -> int:
        return self._index.ntotal if self._index else 0

    def clear(self) -> None:
        self._index = None
        self._chunks = []


# ── Chroma ─────────────────────────────────────────────────────────────────────

class ChromaVectorStore(BaseVectorStore):
    """ChromaDB in-memory (or persisted) vector store."""

    name = "chroma"

    def __init__(self, persist_dir: Optional[str] = None, collection_name: str = "rag_bench"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    def _ensure_collection(self):
        if self._collection is not None:
            return
        try:
            import chromadb
            if self.persist_dir:
                self._client = chromadb.PersistentClient(path=self.persist_dir)
            else:
                self._client = chromadb.EphemeralClient()
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.debug(f"Chroma collection ready: {self.collection_name}")
        except ImportError:
            logger.error("chromadb not installed. Run: pip install chromadb")
            raise

    def index(self, chunks: list[str], embeddings: np.ndarray) -> None:
        self._ensure_collection()
        
        batch_size = 5000  # ChromaDB/SQLite safe limit is ~5461
        num_chunks = len(chunks)
        
        for i in range(0, num_chunks, batch_size):
            end_idx = min(i + batch_size, num_chunks)
            batch_chunks = chunks[i:end_idx]
            batch_embeddings = embeddings[i:end_idx].tolist()
            batch_ids = [str(uuid.uuid4()) for _ in range(len(batch_chunks))]
            
            self._collection.add(
                ids=batch_ids,
                documents=batch_chunks,
                embeddings=batch_embeddings,
            )
        
        logger.debug(f"Chroma: indexed {len(chunks)} chunks in { (num_chunks + batch_size - 1) // batch_size } batches")

    def query(self, embedding: np.ndarray, top_k: int = 5) -> list[str]:
        self._ensure_collection()
        if self._collection.count() == 0:
            return []
        k = min(top_k, self._collection.count())
        result = self._collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=k,
        )
        return result["documents"][0] if result["documents"] else []

    def get_chunk_count(self) -> int:
        self._ensure_collection()
        return self._collection.count()

    def clear(self) -> None:
        if self._client and self._collection:
            self._client.delete_collection(self.collection_name)
            self._collection = None


# ── Qdrant ─────────────────────────────────────────────────────────────────────

class QdrantVectorStore(BaseVectorStore):
    """Qdrant in-memory vector store."""

    name = "qdrant"

    def __init__(self, collection_name: str = "rag_bench", location: str = ":memory:"):
        self.collection_name = collection_name
        self.location = location
        self._client = None
        self._collection_created = False
        self._dimension: Optional[int] = None
        self._id_counter = 0

    def _ensure_client(self, dimension: Optional[int] = None):
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
                if self.location == ":memory:":
                    self._client = QdrantClient(location=":memory:")
                else:
                    self._client = QdrantClient(url=self.location)
                logger.debug("Qdrant client initialised (in-memory)")
            except ImportError:
                logger.error("qdrant-client not installed. Run: pip install qdrant-client")
                raise

        if not self._collection_created and dimension:
            from qdrant_client.models import Distance, VectorParams
            self._client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )
            self._collection_created = True
            self._dimension = dimension

    def index(self, chunks: list[str], embeddings: np.ndarray) -> None:
        dim = embeddings.shape[1]
        self._ensure_client(dimension=dim)
        from qdrant_client.models import PointStruct
        
        batch_size = 1000  # Qdrant recommendation for batched upserts
        num_chunks = len(chunks)
        
        for i in range(0, num_chunks, batch_size):
            end_idx = min(i + batch_size, num_chunks)
            batch_chunks = chunks[i:end_idx]
            batch_embeddings = embeddings[i:end_idx]
            
            points = []
            for j, (chunk, vec) in enumerate(zip(batch_chunks, batch_embeddings)):
                points.append(
                    PointStruct(
                        id=self._id_counter + i + j,
                        vector=vec.tolist(),
                        payload={"text": chunk},
                    )
                )
            self._client.upsert(collection_name=self.collection_name, points=points)
            
        self._id_counter += num_chunks
        logger.debug(f"Qdrant: indexed {len(chunks)} chunks in { (num_chunks + batch_size - 1) // batch_size } batches (total={self._id_counter})")

    def query(self, embedding: np.ndarray, top_k: int = 5) -> list[str]:
        self._ensure_client()
        if not self._collection_created or self._id_counter == 0:
            return []
        k = min(top_k, self._id_counter)
        results = self._client.search(
            collection_name=self.collection_name,
            query_vector=embedding.tolist(),
            limit=k,
        )
        return [r.payload["text"] for r in results]

    def get_chunk_count(self) -> int:
        return self._id_counter

    def clear(self) -> None:
        if self._client and self._collection_created:
            self._client.delete_collection(self.collection_name)
            self._collection_created = False
            self._id_counter = 0


# ── Registry & Factory ─────────────────────────────────────────────────────────

_STORE_REGISTRY: dict[str, type[BaseVectorStore]] = {
    "faiss":  FAISSVectorStore,
    "chroma": ChromaVectorStore,
    "qdrant": QdrantVectorStore,
}


def get_vector_store(name: str, **kwargs) -> BaseVectorStore:
    """
    Instantiate a vector store by name.

    Parameters
    ----------
    name   : 'faiss' | 'chroma' | 'qdrant'
    **kwargs: forwarded to the store constructor
    """
    name = name.lower()
    if name not in _STORE_REGISTRY:
        raise ValueError(
            f"Unknown vector store '{name}'. Available: {list(_STORE_REGISTRY.keys())}"
        )
    return _STORE_REGISTRY[name](**kwargs)


def register_vector_store(name: str, cls: type[BaseVectorStore]) -> None:
    _STORE_REGISTRY[name] = cls
