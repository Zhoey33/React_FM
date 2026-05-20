"""ExpeL Insight Memory Store.

Stores abstract insight rules extracted by ExpeL's cross-trajectory comparison.
Retrieval is by task_type matching (ExpeL's original design) with optional
embedding-based similarity for ranking.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class InsightEntry:
    insight_id: int
    rule: str
    task_type: str
    source_env_idx: int = -1
    created_at: str = ""
    embedding: np.ndarray | None = None

    def to_prompt_str(self) -> str:
        return f"- {self.rule}"

    def to_dict(self) -> dict:
        return {
            "insight_id": self.insight_id,
            "rule": self.rule,
            "task_type": self.task_type,
            "source_env_idx": self.source_env_idx,
            "created_at": self.created_at,
        }


class InsightMemoryStore:
    """ExpeL-style insight store. Retrieves by task_type + embedding similarity."""

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        max_insights: int = 200,
        top_k: int = 5,
    ):
        self.embedding_model_name = embedding_model_name
        self.max_insights = max_insights
        self.top_k = top_k
        self._entries: list[InsightEntry] = []
        self._next_id = 0
        self._model = None
        self.total_retrievals = 0
        self.total_hits = 0

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.embedding_model_name)

    def _embed(self, text: str) -> np.ndarray:
        self._load_model()
        return self._model.encode(text, normalize_embeddings=True)

    def add(self, rule: str, task_type: str, source_env_idx: int = -1) -> InsightEntry:
        """Add an insight rule. Deduplicates by checking cosine similarity."""
        embedding = self._embed(rule)

        # Deduplicate: skip if very similar insight already exists
        for existing in self._entries:
            if existing.embedding is not None:
                sim = float(np.dot(embedding, existing.embedding))
                if sim > 0.92:
                    logger.debug(f"Skipping duplicate insight (sim={sim:.3f}): {rule[:60]}")
                    return existing

        entry = InsightEntry(
            insight_id=self._next_id,
            rule=rule,
            task_type=task_type,
            source_env_idx=source_env_idx,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            embedding=embedding,
        )
        self._entries.append(entry)
        self._next_id += 1

        if len(self._entries) > self.max_insights:
            self._entries.pop(0)

        logger.info(f"Insight stored #{entry.insight_id} [{task_type}]: {rule[:80]}")
        return entry

    def retrieve(self, task_type: str, top_k: int | None = None) -> list[InsightEntry]:
        """Retrieve insights for a task type, ranked by embedding similarity."""
        self.total_retrievals += 1
        k = top_k or self.top_k

        # Filter by task_type first (ExpeL's primary retrieval)
        candidates = [e for e in self._entries if e.task_type == task_type]

        # If not enough task-specific insights, include general ones
        if len(candidates) < k:
            others = [e for e in self._entries if e.task_type != task_type]
            candidates.extend(others)

        if not candidates:
            return []

        results = candidates[:k]
        if results:
            self.total_hits += 1
        return results

    def size(self) -> int:
        return len(self._entries)

    def stats(self) -> dict:
        by_type: dict[str, int] = {}
        for e in self._entries:
            by_type[e.task_type] = by_type.get(e.task_type, 0) + 1
        return {
            "total_entries": self.size(),
            "entries_by_task_type": by_type,
            "total_retrievals": self.total_retrievals,
            "total_hits": self.total_hits,
        }

    def save(self, filepath: str) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "next_id": self._next_id,
            "insights": [],
        }
        for entry in self._entries:
            d = entry.to_dict()
            if entry.embedding is not None:
                d["embedding"] = entry.embedding.tolist()
            data["insights"].append(d)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Insights saved: {self.size()} entries -> {filepath}")

    def load(self, filepath: str) -> None:
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"Insight file not found: {filepath}")
            return
        with open(path) as f:
            data = json.load(f)
        self._next_id = data.get("next_id", 0)
        self._entries.clear()
        for d in data.get("insights", []):
            emb = None
            if "embedding" in d:
                emb = np.array(d["embedding"], dtype=np.float32)
            entry = InsightEntry(
                insight_id=d["insight_id"],
                rule=d["rule"],
                task_type=d.get("task_type", ""),
                source_env_idx=d.get("source_env_idx", -1),
                created_at=d.get("created_at", ""),
                embedding=emb,
            )
            self._entries.append(entry)
        logger.info(f"Insights loaded: {self.size()} entries from {filepath}")
