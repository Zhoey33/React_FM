"""Gate module for intervention gate experiments.

Key components:
- checkpoint: CheckpointState, CheckpointStore — save/restore agent state
- features: extract_gate_features — 11-dim feature vector
- progress: compute_progress — benchmark-native progress signals
- canonicalize: canonicalize_store — LLM-based question/repair generation
- branched_rollout: run_branched_rollout — data collection pipeline
- replay_validation: phase_a, phase_b — validation protocol
- train_gate: InterventionGate — XGBoost utility regression gate
"""

from src.gate.checkpoint import CheckpointState, CheckpointStore
from src.gate.features import extract_features_from_checkpoint, extract_features_from_agent_state
from src.gate.train_gate import InterventionGate
from src.gate.branched_rollout import GATE_ARMS, ROLLOUT_ARMS
