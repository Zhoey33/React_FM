"""Two-stage logistic baseline with one-hot categorical features."""

import json
import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.gate.branched_rollout import GATE_ARMS
from src.gate.train_gate_two_stage import derive_two_stage_targets

logger = logging.getLogger(__name__)


def _build_linear_pipeline() -> Pipeline:
    return Pipeline([
        (
            "prep",
            ColumnTransformer(
                transformers=[
                    ("cat", OneHotEncoder(handle_unknown="ignore"), [0, 1]),
                    ("num", StandardScaler(), list(range(2, 11))),
                ]
            ),
        ),
        (
            "clf",
            LogisticRegression(
                max_iter=2000,
                random_state=42,
            ),
        ),
    ])


class TwoStageLogisticGate:
    """Two-stage logistic baseline."""

    def __init__(
        self,
        bypass_threshold: float = 0.0,
        beta: float = 0.3,
        intervene_threshold: float = 0.5,
        margin_eps: float = 0.01,
    ):
        self.stage_a_model = None
        self.stage_b_model = None
        self.bypass_threshold = bypass_threshold
        self.beta = beta
        self.intervene_threshold = intervene_threshold
        self.margin_eps = margin_eps
        self.train_stats = {}
        self.stage_a_positive_rate = 0.0
        self.stage_b_positive_rate = 0.0

    def _build_stage_datasets(self, data: list[dict]) -> tuple[dict, dict]:
        X_a = []
        y_a = []
        X_b = []
        y_b = []

        for item in data:
            target = derive_two_stage_targets(item)
            features = np.asarray(item["features"], dtype=np.float32)

            if abs(float(target["stage_a_margin"])) > self.margin_eps:
                X_a.append(features)
                y_a.append(int(target["stage_a_label"]))

            if (
                float(target["stage_a_margin"]) > self.margin_eps
                and abs(float(target["stage_b_margin"])) > self.margin_eps
            ):
                X_b.append(features)
                y_b.append(int(target["stage_b_label"]))

        return (
            {"X": np.array(X_a, dtype=np.float32), "y": np.array(y_a, dtype=np.int32)},
            {"X": np.array(X_b, dtype=np.float32), "y": np.array(y_b, dtype=np.int32)},
        )

    def train(self, train_data: list[dict], test_data: list[dict] | None = None) -> dict:
        stage_a_train, stage_b_train = self._build_stage_datasets(train_data)
        if len(stage_a_train["X"]) == 0 or len(np.unique(stage_a_train["y"])) < 2:
            raise ValueError("Stage A has insufficient class coverage after margin filtering")
        if len(stage_b_train["X"]) == 0 or len(np.unique(stage_b_train["y"])) < 2:
            raise ValueError("Stage B has insufficient class coverage after margin filtering")

        self.stage_a_positive_rate = float(np.mean(stage_a_train["y"]))
        self.stage_b_positive_rate = float(np.mean(stage_b_train["y"]))

        self.stage_a_model = _build_linear_pipeline()
        self.stage_b_model = _build_linear_pipeline()
        self.stage_a_model.fit(stage_a_train["X"], stage_a_train["y"])
        self.stage_b_model.fit(stage_b_train["X"], stage_b_train["y"])

        stage_a_train_acc = float(np.mean(
            self.stage_a_model.predict(stage_a_train["X"]) == stage_a_train["y"]
        ))
        stage_b_train_acc = float(np.mean(
            self.stage_b_model.predict(stage_b_train["X"]) == stage_b_train["y"]
        ))

        stats = {
            "n_train_checkpoints": len(train_data),
            "n_stage_a_train": int(len(stage_a_train["X"])),
            "n_stage_b_train": int(len(stage_b_train["X"])),
            "intervene_threshold": round(self.intervene_threshold, 4),
            "stage_a_train_positive_rate": round(self.stage_a_positive_rate, 4),
            "stage_b_train_repair_rate": round(self.stage_b_positive_rate, 4),
            "stage_a_train_accuracy": round(stage_a_train_acc, 4),
            "stage_b_train_accuracy": round(stage_b_train_acc, 4),
        }
        if test_data:
            stats.update(self._evaluate(test_data))
        self.train_stats = stats
        return stats

    def _predict_stage_a_prob(self, features: np.ndarray) -> float:
        row = np.asarray(features, dtype=np.float32).reshape(1, -1)
        return float(self.stage_a_model.predict_proba(row)[0, 1])

    def _predict_stage_b_prob(self, features: np.ndarray) -> float:
        row = np.asarray(features, dtype=np.float32).reshape(1, -1)
        return float(self.stage_b_model.predict_proba(row)[0, 1])

    def predict_details(self, features: np.ndarray) -> dict[str, float | str]:
        p_intervene = self._predict_stage_a_prob(features)
        p_repair = self._predict_stage_b_prob(features)
        pred_arm = "none"
        if p_intervene >= self.intervene_threshold:
            pred_arm = "repair" if p_repair >= 0.5 else "question"
        return {
            "pred_arm": pred_arm,
            "p_intervene": p_intervene,
            "p_none": 1.0 - p_intervene,
            "p_repair_given_intervene": p_repair,
            "p_question_given_intervene": 1.0 - p_repair,
        }

    def _evaluate(self, test_data: list[dict]) -> dict:
        rows = []
        gate_arm_dist = {arm: 0 for arm in GATE_ARMS}

        for item in test_data:
            features = np.asarray(item["features"], dtype=np.float32)
            target = derive_two_stage_targets(item)
            pred = self.predict_details(features)

            true_best = str(target["oracle_arm"])
            pred_best = str(pred["pred_arm"])
            true_intervene = true_best != "none"
            pred_intervene = pred_best != "none"
            gate_arm_dist[pred_best] += 1

            utilities = item["arm_utilities"]
            oracle_utility = float(max(utilities.values()))
            selected_utility = float(utilities[pred_best])
            none_utility = float(utilities["none"])

            rows.append({
                "true_best": true_best,
                "pred_best": pred_best,
                "true_intervene": true_intervene,
                "pred_intervene": pred_intervene,
                "selected_utility": selected_utility,
                "oracle_utility": oracle_utility,
                "none_utility": none_utility,
            })

        if not rows:
            return {
                "n_test_checkpoints": 0,
                "two_stage_accuracy": 0.0,
                "binary_intervene_accuracy": 0.0,
                "intervene_precision": 0.0,
                "intervene_recall": 0.0,
                "question_vs_repair_accuracy_on_oracle_intervene": 0.0,
                "oracle_intervene_rate": 0.0,
                "predicted_intervene_rate": 0.0,
                "avg_selected_utility": 0.0,
                "avg_oracle_utility": 0.0,
                "avg_none_utility": 0.0,
                "avg_regret": 0.0,
                "avg_gain_over_none": 0.0,
                "gate_arm_distribution": gate_arm_dist,
            }

        n = len(rows)
        tp = sum(r["pred_intervene"] and r["true_intervene"] for r in rows)
        fp = sum(r["pred_intervene"] and not r["true_intervene"] for r in rows)
        fn = sum((not r["pred_intervene"]) and r["true_intervene"] for r in rows)
        intervene_subset = [r for r in rows if r["true_intervene"]]

        return {
            "n_test_checkpoints": n,
            "two_stage_accuracy": round(sum(r["pred_best"] == r["true_best"] for r in rows) / n, 4),
            "binary_intervene_accuracy": round(
                sum(r["pred_intervene"] == r["true_intervene"] for r in rows) / n, 4
            ),
            "intervene_precision": round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0,
            "intervene_recall": round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0,
            "question_vs_repair_accuracy_on_oracle_intervene": round(
                sum(r["pred_best"] == r["true_best"] for r in intervene_subset) / len(intervene_subset),
                4,
            ) if intervene_subset else 0.0,
            "oracle_intervene_rate": round(sum(r["true_intervene"] for r in rows) / n, 4),
            "predicted_intervene_rate": round(sum(r["pred_intervene"] for r in rows) / n, 4),
            "avg_selected_utility": round(float(np.mean([r["selected_utility"] for r in rows])), 4),
            "avg_oracle_utility": round(float(np.mean([r["oracle_utility"] for r in rows])), 4),
            "avg_none_utility": round(float(np.mean([r["none_utility"] for r in rows])), 4),
            "avg_regret": round(
                float(np.mean([r["oracle_utility"] - r["selected_utility"] for r in rows])),
                4,
            ),
            "avg_gain_over_none": round(
                float(np.mean([r["selected_utility"] - r["none_utility"] for r in rows])),
                4,
            ),
            "gate_arm_distribution": gate_arm_dist,
        }

    def save(self, filepath: str) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        config = {
            "bypass_threshold": self.bypass_threshold,
            "beta": self.beta,
            "intervene_threshold": self.intervene_threshold,
            "margin_eps": self.margin_eps,
            "train_stats": self.train_stats,
            "stage_a_positive_rate": self.stage_a_positive_rate,
            "stage_b_positive_rate": self.stage_b_positive_rate,
        }
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(config, f, indent=2)
        payload = {
            "stage_a_model": self.stage_a_model,
            "stage_b_model": self.stage_b_model,
        }
        with open(path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(payload, f)
