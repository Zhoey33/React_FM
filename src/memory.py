"""Failure Memory Store with hybrid BM25 + Embedding retrieval (RRF fusion).

Storage is bucketed by a configurable scope:
- ``env_idx`` (default): legacy per-episode/per-env isolation
- ``task_type``: share memories across variations of the same task
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
class RetrievalResult:
    """Result of a memory retrieval with RRF scores for the gate."""
    entries: list  # list[FailureMemoryEntry]
    rrf_scores: list[float]  # RRF score per returned entry (same order)
    candidate_count: int = 0  # number of eligible candidates considered

    @property
    def top1_score(self) -> float:
        """Top-1 RRF score (gate feature: retrieval_rrf_score)."""
        return self.rrf_scores[0] if self.rrf_scores else 0.0

    @property
    def margin(self) -> float:
        """Top-1 minus top-2 RRF score (gate feature: retrieval_margin)."""
        if len(self.rrf_scores) >= 2:
            return self.rrf_scores[0] - self.rrf_scores[1]
        return self.rrf_scores[0] if self.rrf_scores else 0.0


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
    # Gate-related fields (populated by canonicalization pipeline)
    question_text: str = ""  # LLM-canonicalized diagnostic question (no answer revealed)
    repair_text: str = ""    # LLM-canonicalized full repair instruction
    # Tiered repair fields (v2 canonicalization — strategy/tactic/action levels)
    repair_strategy: str = ""  # L1: high-level goal/direction guidance
    repair_tactic: str = ""    # L2: multi-step plan for next actions
    repair_action: str = ""    # L3: exact corrective command(s)
    # ScienceWorld provenance fields for memory-quality analysis.
    failure_step: int | None = None
    failure_type: str = ""
    detector_source: str = ""
    score_before_action: float | None = None
    score_after_action: float | None = None
    score_delta: float | None = None
    source_episode_success: bool | None = None
    source_episode_score: float | None = None
    confidence_score: float | None = None

    def to_prompt_str(self) -> str:
        return (
            f"Situation: Tried '{self.failure_action}' but got '{self.failure_observation}'\n"
            f"Solution: {self.solution_action}"
        )

    def get_repair_display(self) -> str:
        """Return the best available repair text for prompt injection.

        Priority: tiered repair (strategy/tactic/action) > repair_text > solution_action.
        Tiered format gives the agent strategic context, not just action-level fixes.
        """
        if self.repair_strategy:
            parts = []
            parts.append(f"[Strategy] {self.repair_strategy}")
            if self.repair_tactic:
                parts.append(f"[Plan] {self.repair_tactic}")
            if self.repair_action:
                parts.append(f"[Next action] {self.repair_action}")
            return "\n".join(parts)
        if self.repair_text:
            return self.repair_text
        return self.solution_action

    def to_dict(self) -> dict:
        d = {
            "memory_id": self.memory_id,
            "failure_action": self.failure_action,
            "failure_observation": self.failure_observation,
            "solution_action": self.solution_action,
            "task_type": self.task_type,
            "env_idx": self.env_idx,
            "created_at": self.created_at,
        }
        # Only include gate fields if they are populated
        if self.question_text:
            d["question_text"] = self.question_text
        if self.repair_text:
            d["repair_text"] = self.repair_text
        if self.repair_strategy:
            d["repair_strategy"] = self.repair_strategy
        if self.repair_tactic:
            d["repair_tactic"] = self.repair_tactic
        if self.repair_action:
            d["repair_action"] = self.repair_action
        if self.failure_step is not None:
            d["failure_step"] = self.failure_step
        if self.failure_type:
            d["failure_type"] = self.failure_type
        if self.detector_source:
            d["detector_source"] = self.detector_source
        if self.score_before_action is not None:
            d["score_before_action"] = self.score_before_action
        if self.score_after_action is not None:
            d["score_after_action"] = self.score_after_action
        if self.score_delta is not None:
            d["score_delta"] = self.score_delta
        if self.source_episode_success is not None:
            d["source_episode_success"] = self.source_episode_success
        if self.source_episode_score is not None:
            d["source_episode_score"] = self.source_episode_score
        if self.confidence_score is not None:
            d["confidence_score"] = self.confidence_score
        return d


class FailureMemoryStore:
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        max_entries: int = 500,
        top_k: int = 3,
        rrf_k: int = 60,
        min_score: float = 0.0,
        retrieval_mode: str = "hybrid",
        scope: str = "env_idx",
    ):
        if scope not in {"env_idx", "task_type"}:
            raise ValueError(f"Unsupported memory scope: {scope}")
        self.embedding_model_name = embedding_model_name
        self.max_entries = max_entries  # per-bucket max
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.min_score = min_score
        self.retrieval_mode = retrieval_mode  # "hybrid", "bm25_only", "embedding_only", "random"
        self.scope = scope
        self._env_entries: dict[object, list[FailureMemoryEntry]] = {}
        self._next_id = 0
        self._model = None
        # tracking
        self.total_retrievals = 0
        self.total_hits = 0  # retrievals that returned >=1 result

    def _bucket_key(self, task_type: str = "", env_idx: int = 0):
        """Resolve the storage bucket key for the configured scope."""
        if self.scope == "task_type":
            return task_type or "__unknown_task_type__"
        return env_idx

    def _bucket_label(self) -> str:
        return "task_type" if self.scope == "task_type" else "env_idx"

    def get_bucket_entries(
        self,
        task_type: str = "",
        env_idx: int = 0,
        cross_env: bool = False,
    ) -> list[FailureMemoryEntry]:
        """Return entries from the active bucket, or all buckets if cross_env=True."""
        if cross_env:
            entries: list[FailureMemoryEntry] = []
            for bucket_entries in self._env_entries.values():
                entries.extend(bucket_entries)
            return entries
        bucket_key = self._bucket_key(task_type=task_type, env_idx=env_idx)
        return list(self._env_entries.get(bucket_key, []))

    def entry_count(
        self,
        task_type: str = "",
        env_idx: int = 0,
        cross_env: bool = False,
    ) -> int:
        return len(self.get_bucket_entries(task_type=task_type, env_idx=env_idx, cross_env=cross_env))

    def find_entry_by_id(self, memory_id: int) -> FailureMemoryEntry | None:
        for entries in self._env_entries.values():
            for entry in entries:
                if entry.memory_id == memory_id:
                    return entry
        return None

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

    @staticmethod
    def _build_query_text(action: str, observation: str, failure_type: str = "") -> str:
        parts = []
        if failure_type:
            parts.append(f"Failure type: {failure_type}")
        parts.extend([f"Action: {action}", f"Observation: {observation}"])
        return " | ".join(parts)

    @classmethod
    def _build_memory_text(cls, entry: FailureMemoryEntry) -> str:
        return cls._build_query_text(
            entry.failure_action,
            entry.failure_observation,
            failure_type=entry.failure_type,
        )

    def add(
        self,
        failure_action: str,
        failure_observation: str,
        solution_action: str,
        task_type: str = "",
        env_idx: int = 0,
        question_text: str = "",
        repair_strategy: str = "",
        repair_tactic: str = "",
        repair_action: str = "",
        failure_step: int | None = None,
        failure_type: str = "",
        detector_source: str = "",
        score_before_action: float | None = None,
        score_after_action: float | None = None,
        score_delta: float | None = None,
        source_episode_success: bool | None = None,
        source_episode_score: float | None = None,
        confidence_score: float | None = None,
    ) -> FailureMemoryEntry:
        query_text = self._build_query_text(
            failure_action,
            failure_observation,
            failure_type=failure_type,
        )
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
            question_text=question_text,
            repair_strategy=repair_strategy,
            repair_tactic=repair_tactic,
            repair_action=repair_action,
            failure_step=failure_step,
            failure_type=failure_type,
            detector_source=detector_source,
            score_before_action=score_before_action,
            score_after_action=score_after_action,
            score_delta=score_delta,
            source_episode_success=source_episode_success,
            source_episode_score=source_episode_score,
            confidence_score=confidence_score,
        )

        bucket_key = self._bucket_key(task_type=task_type, env_idx=env_idx)
        if bucket_key not in self._env_entries:
            self._env_entries[bucket_key] = []
        self._env_entries[bucket_key].append(entry)
        self._next_id += 1

        if len(self._env_entries[bucket_key]) > self.max_entries:
            self._env_entries[bucket_key].pop(0)

        logger.info(
            f"Memory stored #{entry.memory_id} [{self._bucket_label()}={bucket_key}]: "
            f"[{task_type}] {failure_action[:50]}... → {solution_action[:50]}..."
        )
        return entry

    def retrieve(
        self,
        query_action: str,
        query_observation: str,
        task_type: str = "",
        top_k: int | None = None,
        candidate_k: int | None = None,
        env_idx: int = 0,
        cross_env: bool = False,
        return_scores: bool = False,
        query_failure_type: str = "",
    ) -> "list[FailureMemoryEntry] | RetrievalResult":
        """Retrieve relevant memories.

        Args:
            return_scores: If True, return RetrievalResult with RRF scores
                           (needed by the intervention gate). Default False
                           for backward compatibility.
        """
        self.total_retrievals += 1
        k = candidate_k or top_k or self.top_k

        candidates = self.get_bucket_entries(
            task_type=task_type,
            env_idx=env_idx,
            cross_env=cross_env,
        )
        if not candidates:
            return RetrievalResult([], []) if return_scores else []

        # Filter by task_type if provided
        if task_type:
            typed = [e for e in candidates if e.task_type == task_type]
            if typed:
                candidates = typed
        candidate_count = len(candidates)

        # Random retrieval mode
        if self.retrieval_mode == "random":
            import random
            results = random.sample(candidates, min(k, len(candidates)))
            if results:
                self.total_hits += 1
            if return_scores:
                # Assign uniform pseudo-scores for random mode
                return RetrievalResult(
                    results,
                    [1.0 / len(candidates)] * len(results),
                    candidate_count=candidate_count,
                )
            return results

        query_text = self._build_query_text(
            query_action,
            query_observation,
            failure_type=query_failure_type,
        )

        if self.retrieval_mode == "bm25_only":
            query_tokens = self._tokenize(query_text)
            corpus = [
                self._tokenize(self._build_memory_text(e))
                for e in candidates
            ]
            bm25 = BM25Okapi(corpus)
            bm25_scores = bm25.get_scores(query_tokens)
            sorted_indices = list(np.argsort(-bm25_scores))
            results = [candidates[i] for i in sorted_indices[:k]]
            scores = [float(bm25_scores[i]) for i in sorted_indices[:k]]
            if results:
                self.total_hits += 1
                logger.debug(
                    f"Memory retrieved {len(results)} entries "
                    f"[{self._bucket_label()}={self._bucket_key(task_type=task_type, env_idx=env_idx)}, mode=bm25_only]"
                )
            if return_scores:
                return RetrievalResult(results, scores, candidate_count=candidate_count)
            return results

        if self.retrieval_mode == "embedding_only":
            query_emb = self._embed(query_text)
            emb_scores = []
            for entry in candidates:
                if entry.embedding is not None:
                    emb_scores.append(float(np.dot(query_emb, entry.embedding)))
                else:
                    emb_scores.append(-1.0)
            sorted_indices = list(np.argsort(-np.array(emb_scores)))
            results = [candidates[i] for i in sorted_indices[:k]]
            scores = [emb_scores[i] for i in sorted_indices[:k]]
            if results:
                self.total_hits += 1
                logger.debug(
                    f"Memory retrieved {len(results)} entries "
                    f"[{self._bucket_label()}={self._bucket_key(task_type=task_type, env_idx=env_idx)}, mode=embedding_only]"
                )
            if return_scores:
                return RetrievalResult(results, scores, candidate_count=candidate_count)
            return results

        # Default: hybrid RRF
        query_tokens = self._tokenize(query_text)
        query_emb = self._embed(query_text)

        # --- BM25 ranking ---
        corpus = [
            self._tokenize(self._build_memory_text(e))
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
        result_rrf = []
        for idx in sorted_indices[:k]:
            if self.min_score > 0 and rrf_scores[idx] < self.min_score:
                break
            results.append(candidates[idx])
            result_rrf.append(rrf_scores[idx])

        if results:
            self.total_hits += 1
            logger.debug(
                f"Memory retrieved {len(results)} entries "
                f"[{self._bucket_label()}={self._bucket_key(task_type=task_type, env_idx=env_idx)}] "
                f"for: {query_action[:50]}... "
                f"top_rrf={result_rrf[0]:.4f}"
            )

        if return_scores:
            return RetrievalResult(results, result_rrf, candidate_count=candidate_count)
        return results

    def save(self, filepath: str) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "next_id": self._next_id,
            "scope": self.scope,
            "buckets": {},
        }
        for bucket_key, entries in self._env_entries.items():
            bucket_data = []
            for entry in entries:
                d = entry.to_dict()
                if entry.embedding is not None:
                    d["embedding"] = entry.embedding.tolist()
                bucket_data.append(d)
            data["buckets"][str(bucket_key)] = bucket_data
        if self.scope == "env_idx":
            data["envs"] = data["buckets"]
        else:
            data["task_types"] = data["buckets"]

        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(
            f"Memory saved: {self.size()} entries ({len(self._env_entries)} {self._bucket_label()} buckets) "
            f"→ {filepath}"
        )

    def load(self, filepath: str) -> None:
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"Memory file not found: {filepath}")
            return

        with open(path) as f:
            data = json.load(f)

        self._next_id = data.get("next_id", 0)
        self._env_entries.clear()

        saved_scope = data.get("scope")
        if saved_scope in {"env_idx", "task_type"}:
            self.scope = saved_scope

        raw_buckets = data.get("buckets")
        if raw_buckets is None:
            if "task_types" in data:
                raw_buckets = data.get("task_types", {})
                self.scope = "task_type"
            else:
                raw_buckets = data.get("envs", {})
                self.scope = "env_idx"

        for bucket_key_str, entries_data in raw_buckets.items():
            bucket_key = int(bucket_key_str) if self.scope == "env_idx" else bucket_key_str
            self._env_entries[bucket_key] = []
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
                    env_idx=d.get("env_idx", bucket_key if self.scope == "env_idx" else 0),
                    created_at=d.get("created_at", ""),
                    embedding=emb,
                    question_text=d.get("question_text", ""),
                    repair_text=d.get("repair_text", ""),
                    repair_strategy=d.get("repair_strategy", ""),
                    repair_tactic=d.get("repair_tactic", ""),
                    repair_action=d.get("repair_action", ""),
                    failure_step=d.get("failure_step"),
                    failure_type=d.get("failure_type", ""),
                    detector_source=d.get("detector_source", ""),
                    score_before_action=d.get("score_before_action"),
                    score_after_action=d.get("score_after_action"),
                    score_delta=d.get("score_delta"),
                    source_episode_success=d.get("source_episode_success"),
                    source_episode_score=d.get("source_episode_score"),
                    confidence_score=d.get("confidence_score"),
                )
                self._env_entries[bucket_key].append(entry)

        logger.info(
            f"Memory loaded: {self.size()} entries ({len(self._env_entries)} {self._bucket_label()} buckets) "
            f"from {filepath}"
        )

    def get_all(
        self,
        env_idx: int = 0,
        task_type: str = "",
        cross_env: bool = False,
    ) -> list[FailureMemoryEntry]:
        """Return all entries for the active bucket, or all buckets if cross_env=True."""
        return self.get_bucket_entries(task_type=task_type, env_idx=env_idx, cross_env=cross_env)

    def get_env_ids(self) -> set:
        """Return set of active bucket keys that have memory entries."""
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
