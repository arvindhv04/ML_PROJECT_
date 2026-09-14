"""
vector_store.py
---------------
Member 3 — local ChromaDB pattern store for VishingGuard-ZSL.

Stores KorCCViD transcripts (vishing AND benign, with labels) so the RAG
verifier can check whether nearest neighbours support Member 2's verdict.

Embeddings: sentence-transformers all-MiniLM-L6-v2 (local, free).
Override with EMBEDDING_MODEL if you want a Korean-friendlier model such as
paraphrase-multilingual-MiniLM-L12-v2 (also local/free).

Public API:
    store = VectorStore()                         # persistent local DB
    store.populate_from_csv("final_dataset_m16.csv")  # full labeled dataset
    store.add_pattern(doc_id, text, label, meta)
    store.query_similar(transcript, n_results=3)

CLI:
    python vector_store.py --csv final_dataset_m16.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence

logger = logging.getLogger("vishingguard.vector_store")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO))

DEFAULT_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
DEFAULT_COLLECTION = os.getenv("CHROMA_COLLECTION", "vishingguard_patterns")
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEFAULT_CSV = os.getenv("KORCCVID_CSV", "final_dataset_m16.csv")
LABEL_VISHING_VALUE = os.getenv("LABEL_VISHING_VALUE", "1")
BATCH_SIZE = 64


class EmbeddingFunction(Protocol):
    def __call__(self, input: Sequence[str]) -> List[List[float]]:  # noqa: A003 — Chroma API name
        ...

    def name(self) -> str: ...


class SentenceTransformerEmbedder:
    """Lazy-loaded local embedder. First call downloads the model weights."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model %s (local, first run may download weights)", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def __call__(self, input: Sequence[str]) -> List[List[float]]:  # noqa: A003
        vectors = self._load().encode(list(input), normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vectors]

    def name(self) -> str:
        return f"st:{self.model_name}"


class HashingEmbedder:
    """Deterministic 384-d embedder for unit tests. No downloads, no torch."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def __call__(self, input: Sequence[str]) -> List[List[float]]:  # noqa: A003
        import hashlib
        import math
        import re

        out: List[List[float]] = []
        for text in input:
            vec = [0.0] * self.dim
            folded = text.lower()
            tokens = folded.split()
            # Korean has few spaces; character n-grams keep similar transcripts close.
            compact = re.sub(r"\s+", "", folded)
            for n in (2, 3):
                tokens.extend(compact[i:i + n] for i in range(max(0, len(compact) - n + 1)))
            if not tokens:
                tokens = ["_empty_"]
            for tok in tokens:
                digest = hashlib.sha256(tok.encode("utf-8")).digest()
                # Spread each token across a few dimensions.
                for i in range(0, len(digest) - 3, 4):
                    idx = int.from_bytes(digest[i:i + 2], "little") % self.dim
                    sign = 1.0 if digest[i + 2] % 2 == 0 else -1.0
                    mag = (digest[i + 3] + 1) / 256.0
                    vec[idx] += sign * mag
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            out.append([x / norm for x in vec])
        return out

    def name(self) -> str:
        return "hashing-384"


def load_korccvid_rows(csv_path: str) -> List[Dict[str, str]]:
    """Load Member 2's KorCCViD CSV (utf-8-sig so the BOM on `id` is stripped)."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"KorCCViD CSV not found: {csv_path}")
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"No rows in {csv_path}")
    missing = {"id", "transcript", "label"} - set(rows[0].keys())
    if missing:
        raise ValueError(f"{csv_path} missing columns {missing}; got {list(rows[0].keys())}")
    return rows


def is_vishing_label(label: str) -> bool:
    return str(label).strip() == LABEL_VISHING_VALUE


class VectorStore:
    def __init__(
        self,
        persist_dir: Optional[str] = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_function: Optional[EmbeddingFunction] = None,
        in_memory: bool = False,
    ):
        import chromadb

        self.collection_name = collection_name
        self.embedding_function = embedding_function or SentenceTransformerEmbedder()
        if in_memory or persist_dir is None:
            self.persist_dir = None
            self._client = chromadb.Client()
            logger.info("Opened in-memory Chroma client")
        else:
            self.persist_dir = persist_dir
            os.makedirs(persist_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(path=persist_dir)
            logger.info("Opened persistent Chroma client at %s", persist_dir)

        self.collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "embedding_model": self.embedding_function.name()},
        )

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        self._client.delete_collection(self.collection_name)
        self.collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine", "embedding_model": self.embedding_function.name()},
        )

    def add_pattern(
        self,
        doc_id: str,
        text: str,
        label: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        meta = dict(metadata or {})
        meta["label"] = str(label)
        meta["is_vishing"] = is_vishing_label(label)
        self._upsert_batch([doc_id], [text], [meta])

    def _upsert_batch(self, ids: List[str], documents: List[str], metadatas: List[Dict[str, Any]]) -> None:
        embeddings = self.embedding_function(documents)
        # Chroma metadata values must be scalar.
        clean_meta = []
        for m in metadatas:
            clean_meta.append({k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in m.items()})
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=clean_meta,
            embeddings=embeddings,
        )

    def populate_from_csv(
        self,
        csv_path: str = DEFAULT_CSV,
        vishing_only: bool = False,
        reset: bool = False,
    ) -> Dict[str, int]:
        """
        Embed KorCCViD rows into Chroma.

        vishing_only=True matches the original 'store 695 vishing patterns'
        spec. Default is False: BOTH classes are stored so RAG can return
        benign neighbours (otherwise every query would look like vishing).
        """
        if reset:
            self.reset()

        rows = load_korccvid_rows(csv_path)
        stats = {"read": len(rows), "added": 0, "skipped_empty": 0, "vishing": 0, "benign": 0}

        batch_ids: List[str] = []
        batch_docs: List[str] = []
        batch_meta: List[Dict[str, Any]] = []

        def flush() -> None:
            if not batch_ids:
                return
            self._upsert_batch(batch_ids, batch_docs, batch_meta)
            stats["added"] += len(batch_ids)
            batch_ids.clear()
            batch_docs.clear()
            batch_meta.clear()

        for row in rows:
            text = (row.get("transcript") or "").strip()
            if not text:
                stats["skipped_empty"] += 1
                continue
            label = str(row.get("label", "")).strip()
            vishing = is_vishing_label(label)
            if vishing_only and not vishing:
                continue
            if vishing:
                stats["vishing"] += 1
            else:
                stats["benign"] += 1
            batch_ids.append(str(row["id"]))
            batch_docs.append(text)
            batch_meta.append(
                {
                    "label": label,
                    "is_vishing": vishing,
                    "source": row.get("source") or "",
                    "category": row.get("category") or "",
                }
            )
            if len(batch_ids) >= BATCH_SIZE:
                flush()
                logger.info("Populated %d / %d rows...", stats["added"], stats["read"])
        flush()
        logger.info(
            "Vector store ready: added=%d vishing=%d benign=%d collection_count=%d",
            stats["added"],
            stats["vishing"],
            stats["benign"],
            self.count(),
        )
        return stats

    def query_similar(self, transcript: str, n_results: int = 3) -> List[Dict[str, Any]]:
        if not transcript or not transcript.strip():
            return []
        n_results = max(1, int(n_results))
        available = self.count()
        if available == 0:
            logger.warning("Query against empty collection %s", self.collection_name)
            return []
        k = min(n_results, available)
        embedding = self.embedding_function([transcript])[0]
        raw = self.collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        hits: List[Dict[str, Any]] = []
        ids = (raw.get("ids") or [[]])[0]
        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        dists = (raw.get("distances") or [[]])[0]
        for i, doc_id in enumerate(ids):
            dist = float(dists[i]) if i < len(dists) else 1.0
            # Cosine distance in Chroma is 1 - cosine_similarity (approx).
            similarity = max(0.0, min(1.0, 1.0 - dist))
            meta = metas[i] if i < len(metas) else {}
            label = str(meta.get("label", ""))
            hits.append(
                {
                    "id": doc_id,
                    "document": docs[i] if i < len(docs) else "",
                    "label": label,
                    "is_vishing": bool(meta.get("is_vishing", is_vishing_label(label))),
                    "source": meta.get("source", ""),
                    "category": meta.get("category", ""),
                    "distance": round(dist, 4),
                    "similarity": round(similarity, 4),
                }
            )
        return hits


def get_default_store(in_memory: bool = False) -> VectorStore:
    return VectorStore(in_memory=in_memory)


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate VishingGuard-ZSL ChromaDB from KorCCViD CSV.")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to labeled dataset (default: final_dataset_m16.csv)")
    parser.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR)
    parser.add_argument("--vishing-only", action="store_true", help="Store only label=1 rows")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate the collection first")
    args = parser.parse_args()

    store = VectorStore(persist_dir=args.persist_dir)
    stats = store.populate_from_csv(args.csv, vishing_only=args.vishing_only, reset=args.reset)
    print(stats)
    print(f"collection_count={store.count()} persist_dir={args.persist_dir}")


if __name__ == "__main__":
    main()
