"""Failure Memory Store with hybrid BM25 + Embedding retrieval (RRF fusion).

Per-env storage: each env_idx maintains its own memory entries.
"""

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
    env_idx: int = 0
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
            "env_idx": self.env_idx,
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
        self.max_entries = max_entries  # per-env max
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.min_score = min_score
        self._env_entries: dict[int, list[FailureMemoryEntry]] = {}
        self._next_id = 0
        self._model = None
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

    def add(
        self,
        failure_action: str,
        failure_observation: str,
        solution_action: str,
        task_type: str = "",
        env_idx: int = 0,
    ) -> FailureMemoryEntry:
        query_text = self._build_query_text(failure_action, failure_observation)
        embedding = self._embed(query_text)

        entry = FailureMemoryEntry(
            memory_id=self._next_id,
            failure_action=failure_action,
            failure_observation=failure_observation,
            solution_action=solution_action,
            task_type=task_type,
            env_idx=env_idx,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            embedding=embedding,
        )

        if env_idx not in self._env_entries:
            self._env_entries[env_idx] = []
        self._env_entries[env_idx].append(entry)
        self._next_id += 1

        if len(self._env_entries[env_idx]) > self.max_entries:
            self._env_entries[env_idx].pop(0)

        logger.info(f"Memory stored #{entry.memory_id} [env={env_idx}]: [{task_type}] {failure_action[:50]}... → {solution_action[:50]}...")
        return entry

    def retrieve(
        self,
        query_action: str,
        query_observation: str,
        task_type: str = "",
        top_k: int | None = None,
        env_idx: int = 0,
    ) -> list[FailureMemoryEntry]:
        self.total_retrievals += 1
        k = top_k or self.top_k

        candidates = self._env_entries.get(env_idx, [])
        if not candidates:
            return []

        # Filter by task_type if provided
        if task_type:
            typed = [e for e in candidates if e.task_type == task_type]
            if typed:
                candidates = typed

        query_text = self._build_query_text(query_action, query_observation)
        query_tokens = self._tokenize(query_text)
        query_emb = self._embed(query_text)

        # --- BM25 ranking ---
        corpus = [
            self._tokenize(self._build_query_text(e.failure_action, e.failure_observation))
            for e in candidates
        ]
        bm25 = BM25Okapi(corpus)
        bm25_scores = bm25.get_scores(query_tokens)
        bm25_ranking = np.argsort(-bm25_scores)

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
                f"Memory retrieved {len(results)} entries [env={env_idx}] for: {query_action[:50]}... "
                f"top_rrf={rrf_scores[sorted_indices[0]]:.4f}"
            )

        return results

    def save(self, filepath: str) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "next_id": self._next_id,
            "envs": {},
        }
        for env_idx, entries in self._env_entries.items():
            env_data = []
            for entry in entries:
                d = entry.to_dict()
                if entry.embedding is not None:
                    d["embedding"] = entry.embedding.tolist()
                env_data.append(d)
            data["envs"][str(env_idx)] = env_data

        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Memory saved: {self.size()} entries ({len(self._env_entries)} envs) → {filepath}")

    def load(self, filepath: str) -> None:
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"Memory file not found: {filepath}")
            return

        with open(path) as f:
            data = json.load(f)

        self._next_id = data.get("next_id", 0)
        self._env_entries.clear()

        for env_idx_str, entries_data in data.get("envs", {}).items():
            env_idx = int(env_idx_str)
            self._env_entries[env_idx] = []
            for d in entries_data:
                emb = None
                if "embedding" in d:
                    emb = np.array(d["embedding"], dtype=np.float32)

                entry = FailureMemoryEntry(
                    memory_id=d["memory_id"],
                    failure_action=d["failure_action"],
                    failure_observation=d["failure_observation"],
                    solution_action=d["solution_action"],
                    task_type=d.get("task_type", ""),
                    env_idx=env_idx,
                    created_at=d.get("created_at", ""),
                    embedding=emb,
                )
                self._env_entries[env_idx].append(entry)

        logger.info(f"Memory loaded: {self.size()} entries ({len(self._env_entries)} envs) from {filepath}")

    def get_all(self, env_idx: int = 0) -> list[FailureMemoryEntry]:
        """Return all entries for a given env_idx."""
        return list(self._env_entries.get(env_idx, []))

    def get_env_ids(self) -> set[int]:
        """Return set of env_idxs that have memory entries."""
        return set(self._env_entries.keys())

    def size(self) -> int:
        return sum(len(entries) for entries in self._env_entries.values())

    def stats(self) -> dict:
        by_type: dict[str, int] = {}
        for entries in self._env_entries.values():
            for e in entries:
                by_type[e.task_type] = by_type.get(e.task_type, 0) + 1
        return {
            "total_entries": self.size(),
            "num_envs_with_memory": len(self._env_entries),
            "entries_by_task_type": by_type,
            "total_retrievals": self.total_retrievals,
            "total_hits": self.total_hits,
        }
