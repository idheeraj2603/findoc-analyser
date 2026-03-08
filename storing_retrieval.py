from __future__ import annotations

import hashlib
import logging
from typing import Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import (
    OllamaEmbeddingFunction,
    SentenceTransformerEmbeddingFunction,
)

from chunking import Chunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — override via constants.py in the repo root if needed.
# ---------------------------------------------------------------------------
try:
    from constants import (
        CHROMA_PERSIST_DIR,
        CHROMA_COLLECTION_NAME,
        EMBEDDING_BACKEND,       # "ollama" | "sentence-transformers"
        OLLAMA_EMBED_MODEL,      # e.g. "nomic-embed-text"
        SENTENCE_TRANSFORMER_MODEL,  # e.g. "all-MiniLM-L6-v2"
        OLLAMA_BASE_URL,         # e.g. "http://localhost:11434"
        TOP_K_RESULTS,           # default number of chunks to retrieve
    )
except ImportError:
    CHROMA_PERSIST_DIR = "./chromadb"
    CHROMA_COLLECTION_NAME = "findoc"
    EMBEDDING_BACKEND = "sentence-transformers"
    OLLAMA_EMBED_MODEL = "nomic-embed-text"
    SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
    OLLAMA_BASE_URL = "http://localhost:11434"
    TOP_K_RESULTS = 5


# ---------------------------------------------------------------------------
# Helper — build a stable document ID from chunk content so re-indexing the
# same file is idempotent (ChromaDB upsert deduplicates on ID).
# ---------------------------------------------------------------------------
def _chunk_id(chunk: Chunk, doc_name: str) -> str:
    fingerprint = f"{doc_name}::{chunk.chunk_index}::{chunk.char_start}"
    return hashlib.md5(fingerprint.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Embedding function factory — centralises backend selection.
# ---------------------------------------------------------------------------
def _make_embedding_fn(backend: str = EMBEDDING_BACKEND):
    if backend == "ollama":
        return OllamaEmbeddingFunction(
            url=f"{OLLAMA_BASE_URL}/api/embeddings",
            model_name=OLLAMA_EMBED_MODEL,
        )
    # Default: lightweight local model — no Ollama required.
    return SentenceTransformerEmbeddingFunction(
        model_name=SENTENCE_TRANSFORMER_MODEL
    )


# ---------------------------------------------------------------------------
# Core storage & retrieval class
# ---------------------------------------------------------------------------
class VectorStore:
    """
    Thin wrapper around a ChromaDB collection that handles:
      - Persisting chunk embeddings to disk
      - Upserting chunks (idempotent — safe to re-index the same document)
      - Similarity search with optional metadata filters
      - Listing and deleting indexed documents
    """

    def __init__(
        self,
        persist_dir: str = CHROMA_PERSIST_DIR,
        collection_name: str = CHROMA_COLLECTION_NAME,
        embedding_backend: str = EMBEDDING_BACKEND,
    ) -> None:
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._embed_fn = _make_embedding_fn(embedding_backend)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},   # cosine similarity
        )
        logger.info(
            "VectorStore ready — collection=%r  backend=%r  persist_dir=%r",
            collection_name, embedding_backend, persist_dir,
        )
    

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[Chunk], doc_name: str) -> int:
        """
        Upsert a list of Chunk objects into the vector store.

        Args:
            chunks:   Output of ParagraphAwareChunker.chunk() (or any chunker).
            doc_name: Logical document name used to namespace chunk IDs
                      (typically the uploaded filename).

        Returns:
            Number of chunks upserted.
        """
        if not chunks:
            logger.warning("add_chunks called with empty list — nothing stored.")
            return 0

        ids, documents, metadatas = [], [], []

        for chunk in chunks:
            ids.append(_chunk_id(chunk, doc_name))
            documents.append(chunk.text)
            metadatas.append({
                "doc_name":     doc_name,
                "chunk_index":  chunk.chunk_index,
                "method":       chunk.method,
                "char_start":   chunk.char_start,
                "char_end":     chunk.char_end,
                "section_hint": chunk.section_hint or "",
                # Flatten any caller-supplied extras (must be str/int/float/bool).
                **{k: str(v) for k, v in chunk.extra.items()},
            })

        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("Upserted %d chunks for document %r.", len(chunks), doc_name)
        return len(chunks)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        top_k: int = TOP_K_RESULTS,
        doc_name: Optional[str] = None,
        section_hint: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the most relevant chunks for a natural-language query.

        Args:
            query_text:   The user question or sub-question.
            top_k:        Maximum number of results to return.
            doc_name:     If set, restrict search to a single document.
            section_hint: If set, restrict search to a specific 10-K section
                          (e.g. "Risk Factors", "MD&A").

        Returns:
            List of RetrievedChunk objects sorted by relevance (best first).
        """
        where: dict = {}
        if doc_name:
            where["doc_name"] = doc_name
        if section_hint:
            where["section_hint"] = section_hint

        query_kwargs: dict = dict(
            query_texts=[query_text],
            n_results=min(top_k, self._collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )
        if where:
            query_kwargs["where"] = where

        results = self._collection.query(**query_kwargs)

        retrieved: list[RetrievedChunk] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            retrieved.append(RetrievedChunk(
                text=doc,
                score=1.0 - dist,          # cosine distance → similarity
                doc_name=meta.get("doc_name", ""),
                chunk_index=int(meta.get("chunk_index", -1)),
                section_hint=meta.get("section_hint") or None,
                char_start=int(meta.get("char_start", 0)),
                char_end=int(meta.get("char_end", 0)),
            ))

        logger.debug("Query returned %d chunks for: %r", len(retrieved), query_text)
        return retrieved

    def batch_query(
        self,
        queries: list[str],
        top_k: int = TOP_K_RESULTS,
        doc_name: Optional[str] = None,
        section_hint: Optional[str] = None,
        deduplicate: bool = True,
    ) -> list[RetrievedChunk]:
        """
        Run multiple sub-questions and merge results, optionally deduplicating
        by chunk_index so the same passage isn't fed to the LLM twice.

        Intended for use with the sub-question generator in question_generation.py.
        """
        seen_ids: set[str] = set()
        merged: list[RetrievedChunk] = []

        for q in queries:
            for rc in self.query(q, top_k=top_k, doc_name=doc_name, section_hint=section_hint):
                uid = f"{rc.doc_name}::{rc.chunk_index}"
                if deduplicate and uid in seen_ids:
                    continue
                seen_ids.add(uid)
                merged.append(rc)

        # Re-sort by score descending across all sub-questions.
        merged.sort(key=lambda r: r.score, reverse=True)
        return merged[:top_k]

    # ------------------------------------------------------------------
    # Document management
    # ------------------------------------------------------------------
    
    def list_documents(self) -> list[str]:
        """Return a sorted list of unique document names in the collection."""
        total = self._collection.count()
        if total == 0:
            return []
        all_meta = self._collection.get(include=["metadatas"])["metadatas"]
        return sorted({m.get("doc_name", "") for m in all_meta if m.get("doc_name")})

    def delete_document(self, doc_name: str) -> int:
        """
        Remove all chunks belonging to *doc_name* from the collection.

        Returns:
            Number of chunks deleted.
        """
        ids_to_delete = self._collection.get(
            where={"doc_name": doc_name},
            include=[],
        )["ids"]

        if not ids_to_delete:
            logger.warning("delete_document: no chunks found for %r.", doc_name)
            return 0

        self._collection.delete(ids=ids_to_delete)
        logger.info("Deleted %d chunks for document %r.", len(ids_to_delete), doc_name)
        return len(ids_to_delete)

    def clear_all(self) -> None:
        """Wipe the entire collection. Use with caution."""
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection.name,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning("Collection cleared.")

    def __len__(self) -> int:
        return self._collection.count()



# ---------------------------------------------------------------------------
# Lightweight result dataclass returned by query methods
# ---------------------------------------------------------------------------
from dataclasses import dataclass   # noqa: E402  (after constants to avoid circular)


@dataclass
class RetrievedChunk:
    """A chunk returned from similarity search, enriched with its relevance score."""
    text: str
    score: float            # cosine similarity in [0, 1]; higher = more relevant
    doc_name: str
    chunk_index: int
    section_hint: Optional[str]
    char_start: int
    char_end: int

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return (
            f"RetrievedChunk(score={self.score:.3f}, "
            f"doc={self.doc_name!r}, "
            f"section={self.section_hint!r}, "
            f"preview={preview!r})"
        )

