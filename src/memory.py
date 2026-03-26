"""Failure Memory Store with hybrid BM25 + Embedding retrieval (RRF fusion)."""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


@dataclass
class FailureMemoryEntry:
    memory_id: int
    failure_action: str
    failure_observation: str
    solution_action: str
    task_type: str = ""
    created_at: str = ""
    embedding: np.ndarray | None = None

    def to_prompt_str(self) -> str:
        return (
            f"Situation: Tried '{self.failure_action}' but got '{self.failure_observation}'\n"
            f"Solution: {self.solution_action}"
        )

    def to_dict(self) -> dict:
        return {
            "memory_id": self.memory_id,
            "failure_action": self.failure_action,
            "failure_observation": self.failure_observation,
            "solution_action": self.solution_action,
            "task_type": self.task_type,
            "created_at": self.created_at,
        }


class FailureMemoryStore:
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        max_entries: int = 500,
        top_k: int = 3,
        rrf_k: int = 60,
        min_score: float = 0.0,
    ):
        self.embedding_model_name = embedding_model_name
        self.max_entries = max_entries
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.min_score = min_score
        self.entries: list[FailureMemoryEntry] = []
        self._next_id = 0
        self._model = None
        self._bm25: BM25Okapi | None = None
        self._bm25_corpus: list[list[str]] = []  # tokenized docs for BM25
        # tracking
        self.total_retrievals = 0
        self.total_hits = 0  # retrievals that returned >=1 result

    def _load_model(self):
        if self._model is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.embedding_model_name)
            logger.info("Embedding model loaded.")

    def _embed(self, text: str) -> np.ndarray:
        self._load_model()
        return self._model.encode(text, normalize_embeddings=True)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + lowercase tokenization for BM25."""
        return text.lower().split()

    def _build_query_text(self, action: str, observation: str) -> str:
        return f"{action} | {observation}"

    def _rebuild_bm25(self, candidates: list[FailureMemoryEntry]) -> BM25Okapi:
        """Build BM25 index from candidate entries."""
        corpus = [
            self._tokenize(self._build_query_text(e.failure_action, e.failure_observation))
            for e in candidates
        ]
        return BM25Okapi(corpus)

    def add(
        self,
        failure_action: str,
        failure_observation: str,
        solution_action: str,
        task_type: str = "",
    ) -> FailureMemoryEntry:
        query_text = self._build_query_text(failure_action, failure_observation)
        embedding = self._embed(query_text)

        entry = FailureMemoryEntry(
            memory_id=self._next_id,
            failure_action=failure_action,
            failure_observation=failure_observation,
            solution_action=solution_action,
            task_type=task_type,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            embedding=embedding,
        )
        self.entries.append(entry)
        self._next_id += 1

        if len(self.entries) > self.max_entries:
            self.entries.pop(0)

        # Invalidate BM25 cache (will rebuild on next retrieve)
        self._bm25 = None

        logger.info(f"Memory stored #{entry.memory_id}: [{task_type}] {failure_action[:50]}... → {solution_action[:50]}...")
        return entry

    def retrieve(
        self,
        query_action: str,
        query_observation: str,
        task_type: str = "",
        top_k: int | None = None,
    ) -> list[FailureMemoryEntry]:
        self.total_retrievals += 1
        k = top_k or self.top_k

        if not self.entries:
            return []

        # Filter by task_type if provided
        candidates = self.entries
        if task_type:
            typed = [e for e in self.entries if e.task_type == task_type]
            if typed:
                candidates = typed

        query_text = self._build_query_text(query_action, query_observation)
        query_tokens = self._tokenize(query_text)
        query_emb = self._embed(query_text)

        # --- BM25 ranking ---
        bm25 = self._rebuild_bm25(candidates)
        bm25_scores = bm25.get_scores(query_tokens)
        bm25_ranking = np.argsort(-bm25_scores)  # indices sorted by descending score

        # --- Embedding ranking ---
        emb_scores = []
        for entry in candidates:
            if entry.embedding is not None:
                emb_scores.append(float(np.dot(query_emb, entry.embedding)))
            else:
                emb_scores.append(-1.0)
        emb_ranking = np.argsort(-np.array(emb_scores))

        # --- RRF fusion ---
        rrf_scores: dict[int, float] = {}
        for rank, idx in enumerate(bm25_ranking):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        for rank, idx in enumerate(emb_ranking):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        # Sort by RRF score descending
        sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)

        # Apply min_score filter and take top-k
        results = []
        for idx in sorted_indices[:k]:
            if self.min_score > 0 and rrf_scores[idx] < self.min_score:
                break
            results.append(candidates[idx])

        if results:
            self.total_hits += 1
            logger.debug(
                f"Memory retrieved {len(results)} entries (RRF) for: {query_action[:50]}... "
                f"top_rrf={rrf_scores[sorted_indices[0]]:.4f}"
            )

        return results

    def save(self, filepath: str) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "next_id": self._next_id,
            "entries": [],
        }
        for entry in self.entries:
            d = entry.to_dict()
            if entry.embedding is not None:
                d["embedding"] = entry.embedding.tolist()
            data["entries"].append(d)

        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Memory saved: {len(self.entries)} entries → {filepath}")

    def load(self, filepath: str) -> None:
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"Memory file not found: {filepath}")
            return

        with open(path) as f:
            data = json.load(f)

        self._next_id = data.get("next_id", 0)
        self.entries.clear()

        for d in data.get("entries", []):
            emb = None
            if "embedding" in d:
                emb = np.array(d["embedding"], dtype=np.float32)

            entry = FailureMemoryEntry(
                memory_id=d["memory_id"],
                failure_action=d["failure_action"],
                failure_observation=d["failure_observation"],
                solution_action=d["solution_action"],
                task_type=d.get("task_type", ""),
                created_at=d.get("created_at", ""),
                embedding=emb,
            )
            self.entries.append(entry)

        logger.info(f"Memory loaded: {len(self.entries)} entries from {filepath}")

    def size(self) -> int:
        return len(self.entries)

    def stats(self) -> dict:
        by_type: dict[str, int] = {}
        for e in self.entries:
            by_type[e.task_type] = by_type.get(e.task_type, 0) + 1
        return {
            "total_entries": len(self.entries),
            "entries_by_task_type": by_type,
            "total_retrievals": self.total_retrievals,
            "total_hits": self.total_hits,
        }
