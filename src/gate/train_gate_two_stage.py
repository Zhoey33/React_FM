"""Two-stage intervention gate for ScienceWorld-style arm selection.

Stage A predicts whether to intervene: none vs intervene.
Stage B predicts the intervention type: question vs repair.
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("xgboost not installed — two-stage gate training unavailable")

from src.gate.branched_rollout import GATE_ARMS
from src.gate.features import FEATURE_NAMES


def derive_two_stage_targets(item: dict) -> dict[str, float | int | str]:
    """Derive two-stage margins and labels from arm utilities."""
    utilities = item["arm_utilities"]
    none_u = float(utilities["none"])
    question_u = float(utilities["question"])
    repair_u = float(utilities["repair"])
    intervene_u = max(question_u, repair_u)
    a_margin = intervene_u - none_u
    b_margin = repair_u - question_u
    oracle_arm = max(utilities, key=utilities.get)
    return {
        "none_utility": none_u,
        "question_utility": question_u,
        "repair_utility": repair_u,
        "intervene_utility": intervene_u,
        "stage_a_margin": a_margin,
        "stage_b_margin": b_margin,
        "stage_a_label": int(a_margin > 0.0),
        "stage_b_label": int(b_margin > 0.0),
        "oracle_arm": oracle_arm,
    }


class TwoStageInterventionGate:
    """Two-stage XGBoost gate with a binary intervene decision."""

    def __init__(
        self,
        bypass_threshold: float = 0.0,
        beta: float = 0.3,
        intervene_threshold: float = 0.5,
        margin_eps: float = 0.01,
        stage_a_positive_weight: float = 1.0,
        stage_a_negative_weight: float = 1.0,
        stage_a_margin_weight_alpha: float = 0.0,
    ):
        self.stage_a_model = None
        self.stage_b_model = None
        self.bypass_threshold = bypass_threshold
        self.beta = beta
        self.intervene_threshold = intervene_threshold
        self.margin_eps = margin_eps
        self.stage_a_positive_weight = stage_a_positive_weight
        self.stage_a_negative_weight = stage_a_negative_weight
        self.stage_a_margin_weight_alpha = stage_a_margin_weight_alpha
        self.train_stats = {}
        self.stage_a_positive_rate = 0.0
        self.stage_b_positive_rate = 0.0

    def _build_stage_datasets(self, data: list[dict]) -> tuple[dict, dict]:
        X_a = []
        y_a = []
        m_a = []
        cp_a = []
        X_b = []
        y_b = []
        m_b = []
        cp_b = []

        for item in data:
            target = derive_two_stage_targets(item)
            features = np.asarray(item["features"], dtype=np.float32)
            cp_id = item.get("checkpoint_id", "")

            if abs(float(target["stage_a_margin"])) > self.margin_eps:
                X_a.append(features)
                y_a.append(int(target["stage_a_label"]))
                m_a.append(float(target["stage_a_margin"]))
                cp_a.append(cp_id)

            if (
                float(target["stage_a_margin"]) > self.margin_eps
                and abs(float(target["stage_b_margin"])) > self.margin_eps
            ):
                X_b.append(features)
                y_b.append(int(target["stage_b_label"]))
                m_b.append(float(target["stage_b_margin"]))
                cp_b.append(cp_id)

        return (
            {
                "X": np.array(X_a, dtype=np.float32),
                "y": np.array(y_a, dtype=np.int32),
                "margin": np.array(m_a, dtype=np.float32),
                "checkpoint_ids": cp_a,
            },
            {
                "X": np.array(X_b, dtype=np.float32),
                "y": np.array(y_b, dtype=np.int32),
                "margin": np.array(m_b, dtype=np.float32),
                "checkpoint_ids": cp_b,
            },
        )

    def _build_stage_a_sample_weights(
        self,
        labels: np.ndarray,
        margins: np.ndarray,
    ) -> np.ndarray:
        weights = np.ones(len(labels), dtype=np.float32)
        if len(weights) == 0:
            return weights

        weights *= np.where(
            labels == 1,
            self.stage_a_positive_weight,
            self.stage_a_negative_weight,
        ).astype(np.float32)

        if self.stage_a_margin_weight_alpha > 0.0 and len(margins) > 0:
            abs_margins = np.abs(margins).astype(np.float32)
            mean_abs_margin = float(np.mean(abs_margins))
            if mean_abs_margin > 0.0:
                weights *= 1.0 + (
                    self.stage_a_margin_weight_alpha * (abs_margins / mean_abs_margin)
                )

        return weights

    def train(
        self,
        train_data: list[dict],
        test_data: list[dict] | None = None,
        stage_a_params: dict | None = None,
        stage_b_params: dict | None = None,
    ) -> dict:
        """Train the two-stage gate."""
        if not HAS_XGBOOST:
            raise ImportError("xgboost is required for two-stage gate training")

        stage_a_train, stage_b_train = self._build_stage_datasets(train_data)

        if len(stage_a_train["X"]) == 0 or len(np.unique(stage_a_train["y"])) < 2:
            raise ValueError("Stage A has insufficient class coverage after margin filtering")
        if len(stage_b_train["X"]) == 0 or len(np.unique(stage_b_train["y"])) < 2:
            raise ValueError("Stage B has insufficient class coverage after margin filtering")

        self.stage_a_positive_rate = float(np.mean(stage_a_train["y"]))
        self.stage_b_positive_rate = float(np.mean(stage_b_train["y"]))

        params = {
            "max_depth": 4,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": 42,
        }

        a_params = dict(params)
        b_params = dict(params)
        if stage_a_params:
            a_params.update(stage_a_params)
        if stage_b_params:
            b_params.update(stage_b_params)

        self.stage_a_model = xgb.XGBClassifier(**a_params)
        self.stage_b_model = xgb.XGBClassifier(**b_params)

        stage_a_sample_weights = self._build_stage_a_sample_weights(
            stage_a_train["y"],
            stage_a_train["margin"],
        )

        self.stage_a_model.fit(
            stage_a_train["X"],
            stage_a_train["y"],
            sample_weight=stage_a_sample_weights,
        )
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
            "stage_a_positive_weight": round(self.stage_a_positive_weight, 4),
            "stage_a_negative_weight": round(self.stage_a_negative_weight, 4),
            "stage_a_margin_weight_alpha": round(self.stage_a_margin_weight_alpha, 4),
        }

        if test_data:
            stats.update(self._evaluate(test_data))

        self.train_stats = stats
        logger.info(
            "Two-stage gate training: stage_a=%s rows, stage_b=%s rows",
            len(stage_a_train["X"]),
            len(stage_b_train["X"]),
        )
        return stats

    def _predict_stage_a_prob(self, features: np.ndarray) -> float:
        row = np.asarray(features, dtype=np.float32).reshape(1, -1)
        return float(self.stage_a_model.predict_proba(row)[0, 1])

    def _predict_stage_b_prob(self, features: np.ndarray) -> float:
        row = np.asarray(features, dtype=np.float32).reshape(1, -1)
        return float(self.stage_b_model.predict_proba(row)[0, 1])

    def predict_arm(
        self,
        features: np.ndarray,
        retrieval_rrf_score: float | None = None,
        intervene_threshold: float | None = None,
    ) -> str:
        """Predict final arm with two-stage routing."""
        if retrieval_rrf_score is not None and retrieval_rrf_score < self.bypass_threshold:
            return "none"

        if self.stage_a_model is None or self.stage_b_model is None:
            raise RuntimeError("Two-stage gate not trained yet")

        threshold = self.intervene_threshold if intervene_threshold is None else intervene_threshold
        p_intervene = self._predict_stage_a_prob(features)
        if p_intervene < threshold:
            return "none"

        p_repair = self._predict_stage_b_prob(features)
        return "repair" if p_repair >= 0.5 else "question"

    def predict_details(
        self,
        features: np.ndarray,
        intervene_threshold: float | None = None,
    ) -> dict[str, float | str]:
        """Predict final arm and expose both stage probabilities."""
        threshold = self.intervene_threshold if intervene_threshold is None else intervene_threshold
        p_intervene = self._predict_stage_a_prob(features)
        p_repair = self._predict_stage_b_prob(features)
        pred_arm = "none"
        if p_intervene >= threshold:
            pred_arm = "repair" if p_repair >= 0.5 else "question"
        return {
            "pred_arm": pred_arm,
            "p_intervene": p_intervene,
            "p_none": 1.0 - p_intervene,
            "p_repair_given_intervene": p_repair,
            "p_question_given_intervene": 1.0 - p_repair,
            "intervene_threshold": threshold,
        }

    def evaluate(
        self,
        test_data: list[dict],
        intervene_threshold: float | None = None,
    ) -> dict:
        return self._evaluate(test_data, intervene_threshold=intervene_threshold)

    def _evaluate(
        self,
        test_data: list[dict],
        intervene_threshold: float | None = None,
    ) -> dict:
        rows = []
        gate_arm_dist = {arm: 0 for arm in GATE_ARMS}
        threshold = self.intervene_threshold if intervene_threshold is None else intervene_threshold

        for item in test_data:
            features = np.asarray(item["features"], dtype=np.float32)
            target = derive_two_stage_targets(item)
            pred = self.predict_details(features, intervene_threshold=threshold)

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
            "intervene_threshold": round(float(threshold), 4),
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
        """Save trained models and config."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        config = {
            "bypass_threshold": self.bypass_threshold,
            "beta": self.beta,
            "intervene_threshold": self.intervene_threshold,
            "margin_eps": self.margin_eps,
            "stage_a_positive_weight": self.stage_a_positive_weight,
            "stage_a_negative_weight": self.stage_a_negative_weight,
            "stage_a_margin_weight_alpha": self.stage_a_margin_weight_alpha,
            "train_stats": self.train_stats,
            "stage_a_positive_rate": self.stage_a_positive_rate,
            "stage_b_positive_rate": self.stage_b_positive_rate,
        }
        config_path = path.with_suffix(".json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        model_path = path.with_suffix(".pkl")
        payload = {
            "stage_a_model": self.stage_a_model,
            "stage_b_model": self.stage_b_model,
        }
        with open(model_path, "wb") as f:
            pickle.dump(payload, f)

        logger.info("Two-stage gate saved: %s, %s", model_path, config_path)

    def load(self, filepath: str) -> None:
        """Load trained models and config."""
        path = Path(filepath)

        config_path = path.with_suffix(".json")
        with open(config_path) as f:
            config = json.load(f)
        self.bypass_threshold = config.get("bypass_threshold", 0.0)
        self.beta = config.get("beta", 0.3)
        self.intervene_threshold = config.get("intervene_threshold", 0.5)
        self.margin_eps = config.get("margin_eps", 0.01)
        self.stage_a_positive_weight = config.get("stage_a_positive_weight", 1.0)
        self.stage_a_negative_weight = config.get("stage_a_negative_weight", 1.0)
        self.stage_a_margin_weight_alpha = config.get("stage_a_margin_weight_alpha", 0.0)
        self.train_stats = config.get("train_stats", {})
        self.stage_a_positive_rate = config.get("stage_a_positive_rate", 0.0)
        self.stage_b_positive_rate = config.get("stage_b_positive_rate", 0.0)

        model_path = path.with_suffix(".pkl")
        with open(model_path, "rb") as f:
            payload = pickle.load(f)
        self.stage_a_model = payload["stage_a_model"]
        self.stage_b_model = payload["stage_b_model"]

        logger.info("Two-stage gate loaded: %s", model_path)

    def feature_importance(self) -> dict[str, dict[str, float]]:
        """Get feature importance for both stages."""
        if self.stage_a_model is None or self.stage_b_model is None:
            raise RuntimeError("Two-stage gate not trained yet")

        return {
            "stage_a": {
                name: round(float(imp), 4)
                for name, imp in zip(FEATURE_NAMES, self.stage_a_model.feature_importances_)
            },
            "stage_b": {
                name: round(float(imp), 4)
                for name, imp in zip(FEATURE_NAMES, self.stage_b_model.feature_importances_)
            },
        }
