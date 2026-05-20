"""Two-stage regression gate for ScienceWorld-style arm selection.

Stage A regresses intervene margin: max(question, repair) - none
Stage B regresses repair-vs-question margin: repair - question
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
    logger.warning("xgboost not installed — two-stage regression gate unavailable")

from src.gate.branched_rollout import GATE_ARMS
from src.gate.features import FEATURE_NAMES
from src.gate.train_gate_two_stage import derive_two_stage_targets


class TwoStageRegressionGate:
    """Two-stage XGBoost regressor using margin targets."""

    def __init__(
        self,
        bypass_threshold: float = 0.0,
        beta: float = 0.3,
        margin_eps: float = 0.01,
        stage_a_min_abs_margin: float | None = None,
        intervene_margin_threshold: float = 0.0,
        repair_margin_threshold: float = 0.0,
        stage_a_positive_weight: float = 1.0,
        stage_a_negative_weight: float = 1.0,
        feature_names: list[str] | None = None,
    ):
        self.stage_a_model = None
        self.stage_b_model = None
        self.bypass_threshold = bypass_threshold
        self.beta = beta
        self.margin_eps = margin_eps
        self.stage_a_min_abs_margin = (
            margin_eps if stage_a_min_abs_margin is None else stage_a_min_abs_margin
        )
        self.intervene_margin_threshold = intervene_margin_threshold
        self.repair_margin_threshold = repair_margin_threshold
        self.stage_a_positive_weight = stage_a_positive_weight
        self.stage_a_negative_weight = stage_a_negative_weight
        self.feature_names = feature_names or FEATURE_NAMES
        self.train_stats = {}

    def _build_stage_datasets(self, data: list[dict]) -> tuple[dict, dict]:
        X_a = []
        y_a = []
        X_b = []
        y_b = []

        for item in data:
            target = derive_two_stage_targets(item)
            features = np.asarray(item["features"], dtype=np.float32)

            if abs(float(target["stage_a_margin"])) > self.stage_a_min_abs_margin:
                X_a.append(features)
                y_a.append(float(target["stage_a_margin"]))

            if (
                float(target["stage_a_margin"]) > self.margin_eps
                and abs(float(target["stage_b_margin"])) > self.margin_eps
            ):
                X_b.append(features)
                y_b.append(float(target["stage_b_margin"]))

        return (
            {
                "X": np.array(X_a, dtype=np.float32),
                "y": np.array(y_a, dtype=np.float32),
            },
            {
                "X": np.array(X_b, dtype=np.float32),
                "y": np.array(y_b, dtype=np.float32),
            },
        )

    def _build_stage_a_sample_weights(self, margins: np.ndarray) -> np.ndarray:
        weights = np.ones(len(margins), dtype=np.float32)
        if len(weights) == 0:
            return weights
        weights *= np.where(
            margins >= 0.0,
            self.stage_a_positive_weight,
            self.stage_a_negative_weight,
        ).astype(np.float32)
        return weights

    def train(
        self,
        train_data: list[dict],
        test_data: list[dict] | None = None,
        stage_a_params: dict | None = None,
        stage_b_params: dict | None = None,
    ) -> dict:
        if not HAS_XGBOOST:
            raise ImportError("xgboost is required for two-stage regression gate training")

        stage_a_train, stage_b_train = self._build_stage_datasets(train_data)
        if len(stage_a_train["X"]) == 0:
            raise ValueError("Stage A has no rows after margin filtering")
        if len(stage_b_train["X"]) == 0:
            raise ValueError("Stage B has no rows after margin filtering")

        params = {
            "max_depth": 4,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "random_state": 42,
        }
        a_params = dict(params)
        b_params = dict(params)
        if stage_a_params:
            a_params.update(stage_a_params)
        if stage_b_params:
            b_params.update(stage_b_params)

        self.stage_a_model = xgb.XGBRegressor(**a_params)
        self.stage_b_model = xgb.XGBRegressor(**b_params)
        stage_a_sample_weights = self._build_stage_a_sample_weights(stage_a_train["y"])
        self.stage_a_model.fit(
            stage_a_train["X"],
            stage_a_train["y"],
            sample_weight=stage_a_sample_weights,
        )
        self.stage_b_model.fit(stage_b_train["X"], stage_b_train["y"])

        stage_a_pred = self.stage_a_model.predict(stage_a_train["X"])
        stage_b_pred = self.stage_b_model.predict(stage_b_train["X"])
        stage_a_mae = float(np.mean(np.abs(stage_a_pred - stage_a_train["y"])))
        stage_b_mae = float(np.mean(np.abs(stage_b_pred - stage_b_train["y"])))

        stats = {
            "n_train_checkpoints": len(train_data),
            "n_stage_a_train": int(len(stage_a_train["X"])),
            "n_stage_b_train": int(len(stage_b_train["X"])),
            "stage_a_train_mae": round(stage_a_mae, 4),
            "stage_b_train_mae": round(stage_b_mae, 4),
            "stage_a_min_abs_margin": round(self.stage_a_min_abs_margin, 4),
            "intervene_margin_threshold": round(self.intervene_margin_threshold, 4),
            "repair_margin_threshold": round(self.repair_margin_threshold, 4),
            "stage_a_positive_weight": round(self.stage_a_positive_weight, 4),
            "stage_a_negative_weight": round(self.stage_a_negative_weight, 4),
        }
        if test_data:
            stats.update(self._evaluate(test_data))

        self.train_stats = stats
        logger.info(
            "Two-stage regression training: stage_a=%s rows, stage_b=%s rows",
            len(stage_a_train["X"]),
            len(stage_b_train["X"]),
        )
        return stats

    def _predict_stage_a_margin(self, features: np.ndarray) -> float:
        row = np.asarray(features, dtype=np.float32).reshape(1, -1)
        return float(self.stage_a_model.predict(row)[0])

    def _predict_stage_b_margin(self, features: np.ndarray) -> float:
        row = np.asarray(features, dtype=np.float32).reshape(1, -1)
        return float(self.stage_b_model.predict(row)[0])

    def predict_details(self, features: np.ndarray) -> dict[str, float | str]:
        a_margin = self._predict_stage_a_margin(features)
        b_margin = self._predict_stage_b_margin(features)
        pred_arm = "none"
        if a_margin >= self.intervene_margin_threshold:
            pred_arm = "repair" if b_margin >= self.repair_margin_threshold else "question"
        return {
            "pred_arm": pred_arm,
            "pred_stage_a_margin": a_margin,
            "pred_stage_b_margin": b_margin,
        }

    def predict_arm(
        self,
        features: np.ndarray,
        retrieval_rrf_score: float | None = None,
    ) -> str:
        if retrieval_rrf_score is not None and retrieval_rrf_score < self.bypass_threshold:
            return "none"
        if self.stage_a_model is None or self.stage_b_model is None:
            raise RuntimeError("Two-stage regression gate not trained yet")
        return str(self.predict_details(features)["pred_arm"])

    def _evaluate(self, test_data: list[dict]) -> dict:
        rows = []
        gate_arm_dist = {arm: 0 for arm in GATE_ARMS}
        stage_a_true = []
        stage_a_pred = []
        stage_b_true = []
        stage_b_pred = []

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
            stage_a_true.append(float(target["stage_a_margin"]))
            stage_a_pred.append(float(pred["pred_stage_a_margin"]))
            if true_intervene:
                stage_b_true.append(float(target["stage_b_margin"]))
                stage_b_pred.append(float(pred["pred_stage_b_margin"]))

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
                "stage_a_test_mae": 0.0,
                "stage_b_test_mae": 0.0,
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
            "stage_a_test_mae": round(
                float(np.mean(np.abs(np.asarray(stage_a_pred) - np.asarray(stage_a_true)))),
                4,
            ),
            "stage_b_test_mae": round(
                float(np.mean(np.abs(np.asarray(stage_b_pred) - np.asarray(stage_b_true)))),
                4,
            ) if stage_b_true else 0.0,
            "gate_arm_distribution": gate_arm_dist,
        }

    def save(self, filepath: str) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        config = {
            "bypass_threshold": self.bypass_threshold,
            "beta": self.beta,
            "margin_eps": self.margin_eps,
            "stage_a_min_abs_margin": self.stage_a_min_abs_margin,
            "intervene_margin_threshold": self.intervene_margin_threshold,
            "repair_margin_threshold": self.repair_margin_threshold,
            "stage_a_positive_weight": self.stage_a_positive_weight,
            "stage_a_negative_weight": self.stage_a_negative_weight,
            "feature_names": self.feature_names,
            "train_stats": self.train_stats,
        }
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(config, f, indent=2)
        payload = {
            "stage_a_model": self.stage_a_model,
            "stage_b_model": self.stage_b_model,
        }
        with open(path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(payload, f)

    def feature_importance(self) -> dict[str, dict[str, float]]:
        if self.stage_a_model is None or self.stage_b_model is None:
            raise RuntimeError("Two-stage regression gate not trained yet")
        return {
            "stage_a": {
                name: round(float(imp), 4)
                for name, imp in zip(self.feature_names, self.stage_a_model.feature_importances_)
            },
            "stage_b": {
                name: round(float(imp), 4)
                for name, imp in zip(self.feature_names, self.stage_b_model.feature_importances_)
            },
        }

    def load(self, filepath: str) -> None:
        path = Path(filepath)
        with open(path.with_suffix(".json")) as f:
            config = json.load(f)
        self.bypass_threshold = config.get("bypass_threshold", 0.0)
        self.beta = config.get("beta", 0.3)
        self.margin_eps = config.get("margin_eps", 0.01)
        self.stage_a_min_abs_margin = config.get("stage_a_min_abs_margin", self.margin_eps)
        self.intervene_margin_threshold = config.get("intervene_margin_threshold", 0.0)
        self.repair_margin_threshold = config.get("repair_margin_threshold", 0.0)
        self.stage_a_positive_weight = config.get("stage_a_positive_weight", 1.0)
        self.stage_a_negative_weight = config.get("stage_a_negative_weight", 1.0)
        self.feature_names = config.get("feature_names", FEATURE_NAMES)
        self.train_stats = config.get("train_stats", {})

        with open(path.with_suffix(".pkl"), "rb") as f:
            payload = pickle.load(f)
        self.stage_a_model = payload["stage_a_model"]
        self.stage_b_model = payload["stage_b_model"]
