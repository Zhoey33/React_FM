"""Offline checks for the paired historical-information intervention."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from experiments import compare_memory_direct as pilot


@pytest.mark.parametrize("memory_text", ["", "Past failed action: open jar\nHistorical repair: unlock jar"])
def test_same_repairer_and_guidance_lifetime(tmp_path, memory_text):
    cp = {"checkpoint_id": "case", "episode_id": "episode", "task": "boil",
          "initial": "Boil water", "failure_type": "action_failed",
          "state": {"score": 0, "look": "kitchen", "inventory": "empty"},
          "steps": [{"action": "open jar", "observation": "It is locked.", "score": 0}]}
    (tmp_path / "checkpoints.jsonl").write_text(json.dumps(cp) + "\n")
    args = SimpleNamespace(output=tmp_path, retrieval="hybrid", max_checkpoints=1,
                           workers=1, only_memory_hits=False, replays=1,
                           arms=["direct", "memory"], seed=0, memory_in_repair=True,
                           guidance_until_action=True, branch_steps=3, branch_cny=.12)
    requests = {}

    class FakeCalls:
        def generate(self, prompt, group, cap, repair=False):
            requests.setdefault(group, []).append((repair, prompt))
            if repair:
                return "unlock jar"
            return "think: inspect lock" if len(requests[group]) == 2 else "look at jar"

        def spent(self, group=None):
            return 0

    memory = {"memory_id": "old" if memory_text else None, "text": memory_text}
    with patch.object(pilot, "FailureMemoryStore"), patch.object(pilot, "restore"), \
         patch.object(pilot, "choose_memory", return_value=memory), \
         patch.object(pilot, "snapshot", return_value=cp["state"]), \
         patch.object(pilot, "execute", return_value=("OK", 0, False)):
        pilot.branch(args, FakeCalls(), object())
    direct, remembered = (requests[f"branch/case/0/{arm}"] for arm in args.arms)
    assert sum(repair for repair, _ in direct) == sum(repair for repair, _ in remembered) == 1
    if memory_text:
        extra = "\nHistorical experience, to use only if applicable:\n" + memory_text
        assert remembered[0][1].replace(extra, "") == direct[0][1]
    else:
        assert remembered[0] == direct[0]
    for turns in (direct, remembered):
        assert "Current-state repair advice:" in turns[1][1]
        assert "Current-state repair advice:" in turns[2][1]
        assert "Current-state repair advice:" not in turns[3][1]

    # Frozen retrieval lets concurrent environment workers avoid native model initialization.
    (tmp_path / "branches.jsonl").unlink()
    curated_text = "Past failed action: open jar\nHistorical repair: use the key"
    args.retrieval_plan = tmp_path / "retrieval_plan.json"
    args.retrieval_plan.write_text(json.dumps({"case": {
        "automatic": memory, "curated": {"memory_id": "checked", "text": curated_text}}}))
    args.arms = ["curated"]
    args.environment_actions = 1
    with patch.object(pilot, "FailureMemoryStore", side_effect=AssertionError("Must use frozen plan")), \
         patch.object(pilot, "restore"), patch.object(pilot, "snapshot", return_value=cp["state"]), \
         patch.object(pilot, "execute", return_value=("OK", 0, False)):
        pilot.branch(args, FakeCalls(), object())
    checked = requests["branch/case/0/curated"]
    assert checked[0][1].replace("\nHistorical experience, to use only if applicable:\n" + curated_text, "") == direct[0][1]
    assert len(checked) == 3  # repair, think, one environment action
    assert pilot.read_rows(tmp_path / "branches.jsonl")[0]["status"] == "action_horizon"


def test_shared_actor_interface_in_actual_requests(tmp_path, monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "offline-test-placeholder")
    captured = []

    def completion(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(usage=SimpleNamespace(model_dump=lambda: {
            "prompt_tokens": 10, "completion_tokens": 2}),
            choices=[SimpleNamespace(message=SimpleNamespace(content="look around"),
                                     finish_reason="stop")])

    with patch.object(pilot, "OpenAI") as client:
        client.return_value.chat.completions.create.side_effect = completion
        calls = pilot.Calls(tmp_path, "deepseek-ai/DeepSeek-V4-Flash", 1, True, True)
        calls.generate("state", "actor", .1)
        calls.generate("state", "repair", .1, repair=True)
    guide = pilot.actor_instructions(True)
    assert guide != pilot.SYSTEM_PROMPT_BASE
    assert "teleport to" not in guide
    assert "output only the chosen number" in guide
    assert all(guide in call["messages"][0]["content"] for call in captured)
    assert pilot.actor_instructions(False) == pilot.SYSTEM_PROMPT_BASE
