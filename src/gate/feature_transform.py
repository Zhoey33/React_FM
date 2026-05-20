"""Experimental feature transforms for offline gate training."""

from dataclasses import dataclass

import numpy as np

from src.gate.features import FEATURE_NAMES, FAILURE_TYPE_MAP, TASK_TYPE_MAP


FAILURE_TYPE_DIM = len(FAILURE_TYPE_MAP) + 1
TASK_TYPE_DIM = len(TASK_TYPE_MAP) + 1
FAILURE_TYPE_LABELS = {
    idx: label for label, idx in FAILURE_TYPE_MAP.items()
}
TASK_TYPE_LABELS = {
    idx: label for label, idx in TASK_TYPE_MAP.items()
}


@dataclass(frozen=True)
class FeatureTransformConfig:
    categorical_encoding: str = "int"
    drop_features: tuple[str, ...] = ()


def parse_drop_features(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    values = []
    for chunk in raw.split(","):
        name = chunk.strip()
        if name:
            values.append(name)
    return tuple(values)


def transform_feature_vector(
    base_features: np.ndarray,
    config: FeatureTransformConfig,
) -> tuple[np.ndarray, list[str]]:
    """Transform base 11-d features into an experimental feature vector."""
    base = np.asarray(base_features, dtype=np.float32)
    if base.shape != (len(FEATURE_NAMES),):
        raise ValueError(f"Expected base feature shape {(len(FEATURE_NAMES),)}, got {base.shape}")

    names: list[str] = []
    values: list[float] = []

    if config.categorical_encoding == "int":
        names.extend(FEATURE_NAMES)
        values.extend(float(v) for v in base.tolist())
    elif config.categorical_encoding == "onehot":
        failure_idx = int(base[0])
        task_idx = int(base[1])

        for idx in range(FAILURE_TYPE_DIM):
            label = FAILURE_TYPE_LABELS.get(idx, "<UNK>")
            names.append(f"failure_type={label}")
            values.append(1.0 if failure_idx == idx else 0.0)

        for idx in range(TASK_TYPE_DIM):
            label = TASK_TYPE_LABELS.get(idx, "<UNK>")
            names.append(f"task_type={label}")
            values.append(1.0 if task_idx == idx else 0.0)

        for base_idx in range(2, len(FEATURE_NAMES)):
            names.append(FEATURE_NAMES[base_idx])
            values.append(float(base[base_idx]))
    else:
        raise ValueError(f"Unsupported categorical_encoding={config.categorical_encoding}")

    if config.drop_features:
        drop_set = set(config.drop_features)
        filtered_names = []
        filtered_values = []
        for name, value in zip(names, values):
            root_name = name.split("=", 1)[0]
            if name in drop_set or root_name in drop_set:
                continue
            filtered_names.append(name)
            filtered_values.append(value)
        names = filtered_names
        values = filtered_values

    return np.asarray(values, dtype=np.float32), names


def transform_training_data(
    data: list[dict],
    config: FeatureTransformConfig,
) -> tuple[list[dict], list[str]]:
    transformed = []
    feature_names: list[str] | None = None
    for item in data:
        features, names = transform_feature_vector(item["features"], config)
        if feature_names is None:
            feature_names = names
        transformed.append({**item, "features": features})
    return transformed, (feature_names or [])
