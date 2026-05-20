"""XGBoost gate training: f(features, arm_onehot) → utility.

At inference: π(x) = argmax_{k ∈ {none, question, repair}} f(x, k)

Trains from branched rollout data with replay-averaged utilities.
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
    logger.warning("xgboost not installed — gate training unavailable")

from src.gate.features import FEATURE_NAMES, extract_features_from_checkpoint
from src.gate.branched_rollout import GATE_ARMS, ArmUtility

# Arm one-hot encoding (appended to 11 features → 14 total)
ARM_ONEHOT = {
    "none":     [1, 0, 0],
    "question": [0, 1, 0],
    "repair":   [0, 0, 1],
}

FULL_FEATURE_NAMES = FEATURE_NAMES + ["arm_none", "arm_question", "arm_repair"]


class InterventionGate:
    """Learned intervention gate using XGBoost utility regression."""

    def __init__(
        self,
        bypass_threshold: float = 0.0,
        beta: float = 0.3,
    ):
        self.model = None
        self.bypass_threshold = bypass_threshold  # τ_min for RRF bypass
        self.beta = beta
        self.train_stats = {}
        self.train_target_mean = 0.0

    def _build_feature_row(
        self,
        features: np.ndarray,
        arm: str,
    ) -> np.ndarray:
        """Concatenate 11 checkpoint features + 3 arm one-hot = 14 features."""
        onehot = np.array(ARM_ONEHOT[arm], dtype=np.float32)
        return np.concatenate([features, onehot])

    def train(
        self,
        train_data: list[dict],
        test_data: list[dict] | None = None,
        xgb_params: dict | None = None,
    ) -> dict:
        """Train XGBoost regressor from branched rollout data.

        Args:
            train_data: list of {
                "features": np.ndarray (11,),
                "arm_utilities": {arm: float},  # utility per arm
            }
            test_data: optional held-out data in same format
            xgb_params: XGBoost parameters override

        Returns:
            Training stats dict
        """
        if not HAS_XGBOOST:
            raise ImportError("xgboost is required for gate training")

        # Build training matrix: each (checkpoint, arm) pair is one row
        X_train = []
        y_train = []

        for item in train_data:
            features = item["features"]
            for arm in GATE_ARMS:
                if arm in item["arm_utilities"]:
                    row = self._build_feature_row(features, arm)
                    X_train.append(row)
                    y_train.append(item["arm_utilities"][arm])

        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        self.train_target_mean = float(np.mean(y_train)) if len(y_train) > 0 else 0.0

        logger.info(f"Training gate: {X_train.shape[0]} rows, {X_train.shape[1]} features")

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
        if xgb_params:
            params.update(xgb_params)

        self.model = xgb.XGBRegressor(**params)
        self.model.fit(X_train, y_train)

        # Training metrics
        train_pred = self.model.predict(X_train)
        train_mae = float(np.mean(np.abs(train_pred - y_train)))
        train_mean_pred = float(np.mean(np.abs(self.train_target_mean - y_train)))

        stats = {
            "n_train_rows": len(X_train),
            "n_train_checkpoints": len(train_data),
            "train_mae": round(train_mae, 4),
            "baseline_mae": round(train_mean_pred, 4),  # predict-mean baseline
            "train_better": train_mae < train_mean_pred,
        }

        # Test metrics
        if test_data:
            test_stats = self._evaluate(test_data)
            stats.update(test_stats)

        self.train_stats = stats
        logger.info(f"Gate training: MAE={train_mae:.4f} (baseline={train_mean_pred:.4f})")
        return stats

    def _evaluate(self, test_data: list[dict]) -> dict:
        """Evaluate on held-out data."""
        X_test = []
        y_test = []
        checkpoint_meta = []  # for per-checkpoint analysis

        for item in test_data:
            features = item["features"]
            for arm in GATE_ARMS:
                if arm in item["arm_utilities"]:
                    row = self._build_feature_row(features, arm)
                    X_test.append(row)
                    y_test.append(item["arm_utilities"][arm])
                    checkpoint_meta.append({
                        "cp_id": item.get("checkpoint_id", ""),
                        "arm": arm,
                    })

        X_test = np.array(X_test, dtype=np.float32)
        y_test = np.array(y_test, dtype=np.float32)

        test_pred = self.model.predict(X_test)
        test_mae = float(np.mean(np.abs(test_pred - y_test)))
        test_mean_pred = float(np.mean(np.abs(self.train_target_mean - y_test)))

        # Per-checkpoint gate accuracy
        correct = 0
        total_cps = 0
        gate_arm_dist = {arm: 0 for arm in GATE_ARMS}

        # Group predictions by checkpoint
        cp_preds: dict[str, dict[str, float]] = {}
        cp_trues: dict[str, dict[str, float]] = {}
        for i, meta in enumerate(checkpoint_meta):
            cp_id = meta["cp_id"]
            arm = meta["arm"]
            cp_preds.setdefault(cp_id, {})[arm] = float(test_pred[i])
            cp_trues.setdefault(cp_id, {})[arm] = float(y_test[i])

        for cp_id in cp_preds:
            if len(cp_preds[cp_id]) < len(GATE_ARMS):
                continue
            total_cps += 1
            pred_best = max(cp_preds[cp_id], key=cp_preds[cp_id].get)
            true_best = max(cp_trues[cp_id], key=cp_trues[cp_id].get)
            gate_arm_dist[pred_best] += 1
            if pred_best == true_best:
                correct += 1

        gate_accuracy = correct / total_cps if total_cps > 0 else 0.0

        return {
            "n_test_rows": len(X_test),
            "n_test_checkpoints": total_cps,
            "test_mae": round(test_mae, 4),
            "test_baseline_mae": round(test_mean_pred, 4),
            "test_better": test_mae < test_mean_pred,
            "gate_accuracy": round(gate_accuracy, 4),
            "gate_arm_distribution": gate_arm_dist,
        }

    def predict_arm(
        self,
        features: np.ndarray,
        retrieval_rrf_score: float | None = None,
    ) -> str:
        """Select the best arm for a checkpoint.

        Args:
            features: 11-dim feature vector
            retrieval_rrf_score: if provided and below bypass_threshold,
                                 returns "none" without running the model

        Returns:
            Selected arm: "none", "question", or "repair"
        """
        # Bypass rule: if retrieval confidence too low, don't inject
        if retrieval_rrf_score is not None and retrieval_rrf_score < self.bypass_threshold:
            return "none"

        if self.model is None:
            raise RuntimeError("Gate not trained yet")

        # Predict utility for each arm
        utilities = {}
        for arm in GATE_ARMS:
            row = self._build_feature_row(features, arm).reshape(1, -1)
            utilities[arm] = float(self.model.predict(row)[0])

        # Select arm with highest predicted utility
        best_arm = max(utilities, key=utilities.get)
        return best_arm

    def predict_utilities(self, features: np.ndarray) -> dict[str, float]:
        """Predict utility for all arms (for analysis)."""
        if self.model is None:
            raise RuntimeError("Gate not trained yet")

        utilities = {}
        for arm in GATE_ARMS:
            row = self._build_feature_row(features, arm).reshape(1, -1)
            utilities[arm] = float(self.model.predict(row)[0])
        return utilities

    def save(self, filepath: str) -> None:
        """Save trained model and config."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "bypass_threshold": self.bypass_threshold,
            "beta": self.beta,
            "train_stats": self.train_stats,
            "train_target_mean": self.train_target_mean,
        }
        # Save config as JSON
        config_path = path.with_suffix(".json")
        with open(config_path, "w") as f:
            json.dump(data, f, indent=2)

        # Save model as pickle
        model_path = path.with_suffix(".pkl")
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

        logger.info(f"Gate saved: {model_path}, {config_path}")

    def load(self, filepath: str) -> None:
        """Load trained model and config."""
        path = Path(filepath)

        config_path = path.with_suffix(".json")
        with open(config_path) as f:
            data = json.load(f)
        self.bypass_threshold = data.get("bypass_threshold", 0.0)
        self.beta = data.get("beta", 0.3)
        self.train_stats = data.get("train_stats", {})
        self.train_target_mean = data.get("train_target_mean", 0.0)

        model_path = path.with_suffix(".pkl")
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        logger.info(f"Gate loaded: {model_path}")

    def feature_importance(self) -> dict[str, float]:
        """Get feature importance from trained XGBoost model."""
        if self.model is None:
            raise RuntimeError("Gate not trained yet")

        importances = self.model.feature_importances_
        return {
            name: round(float(imp), 4)
            for name, imp in zip(FULL_FEATURE_NAMES, importances)
        }
