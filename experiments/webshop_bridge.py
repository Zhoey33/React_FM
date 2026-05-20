"""Bridge script for paper-aligned WebShop subprocess communication."""

import sys
import os
import json
import argparse
import warnings
from pathlib import Path
from types import SimpleNamespace

# Ensure Java is available
os.environ["JAVA_HOME"] = "/opt/homebrew/Cellar/openjdk/25.0.2/libexec/openjdk.jdk/Contents/Home"
warnings.filterwarnings("ignore")

WEBSHOP_ROOT = Path("/Users/zhoey/WebShop")

sys.path.insert(0, str(WEBSHOP_ROOT))


def parse_num_products(value: str):
    """Parse product-count CLI value, with None meaning full WebShop."""
    lowered = value.lower()
    if lowered in {"full", "none", "all", "null"}:
        return None
    return int(value)


def normalize_direct_actions(action_info) -> list[str]:
    """Convert direct WebShop action metadata into executable action strings."""
    if isinstance(action_info, list):
        return action_info
    if not isinstance(action_info, dict):
        return []
    if action_info.get("has_search_bar"):
        return ["search[product]"]

    actions = []
    for text in action_info.get("clickables", []):
        if text == "search":
            continue
        actions.append(f"click[{text}]")
    return actions or ["finish"]


def configure_full_data_paths(num_products):
    """Point WebShop modules at full product files when using the full protocol."""
    if num_products is not None:
        return

    full_items = WEBSHOP_ROOT / "data" / "items_shuffle.json"
    full_attrs = WEBSHOP_ROOT / "data" / "items_ins_v2.json"
    full_index = WEBSHOP_ROOT / "search_engine" / "indexes"
    missing = [str(p) for p in (full_items, full_attrs, full_index) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Full WebShop data/index is required for paper-aligned runs. "
            f"Missing: {missing}"
        )

    import web_agent_site.engine.engine as engine_module
    import web_agent_site.envs.web_agent_text_env as text_env_module

    engine_module.DEFAULT_ATTR_PATH = str(full_attrs)
    text_env_module.DEFAULT_FILE_PATH = str(full_items)
    # WebEnv does not expose file_path, so update the constructor default safely.
    defaults = list(text_env_module.WebAgentTextEnv.__init__.__defaults__ or ())
    if len(defaults) >= 2:
        defaults[1] = str(full_items)
        text_env_module.WebAgentTextEnv.__init__.__defaults__ = tuple(defaults)


def make_official_args(args) -> SimpleNamespace:
    """Build the argparse-like object expected by baseline_models.env.WebEnv."""
    return SimpleNamespace(
        state_format=args.observation_mode,
        num=args.num_products,
        human_goals=args.human_goals,
        get_image=0,
        num_prev_obs=0,
        num_prev_actions=0,
        step_limit=args.step_limit,
        click_item_name=True,
        harsh_reward=False,
        go_to_item=False,
        go_to_search=False,
        ban_buy=True,
        extra_search_path="",
    )


def make_env(args):
    """Create either the official wrapper or the legacy direct text environment."""
    configure_full_data_paths(args.num_products)

    if args.wrapper == "official":
        from baseline_models.env import WebEnv

        return WebEnv(make_official_args(args), split=args.split)

    from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

    return WebAgentTextEnv(
        observation_mode=args.observation_mode,
        num_products=args.num_products,
        human_goals=args.human_goals,
    )


def reset_env(env, wrapper: str, session: int) -> dict:
    """Reset WebShop and normalize wrapper-specific outputs."""
    obs = env.reset(session)
    if wrapper == "official":
        obs_text, info = obs
        return {
            "observation": obs_text,
            "actions": info.get("valid", []),
            "goal": info.get("goal", ""),
            "score": info.get("score", 0.0),
        }

    obs_text = obs[0] if isinstance(obs, tuple) else str(obs)
    return {
        "observation": obs_text,
        "actions": normalize_direct_actions(env.get_available_actions()),
    }


def step_env(env, wrapper: str, action: str) -> dict:
    """Step WebShop and normalize official reward scale to 0-1."""
    obs, reward, done, info = env.step(action)
    if info is None:
        info = {}
    if wrapper == "official":
        reward = float(reward) / 10.0
        return {
            "observation": obs,
            "reward": reward,
            "done": done,
            "actions": info.get("valid", []),
            "score": info.get("score", 0.0),
            "raw_info": {
                "goal": info.get("goal", ""),
                "estimate_score": info.get("estimate_score", 0.0),
                "verbose": info.get("verbose", {}),
            },
        }

    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "actions": normalize_direct_actions(env.get_available_actions()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-products", type=parse_num_products, default=None)
    parser.add_argument("--observation-mode", type=str, default="text_rich")
    parser.add_argument("--human-goals", type=int, default=1)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--step-limit", type=int, default=100)
    parser.add_argument("--wrapper", choices=["official", "direct"], default="official")
    args = parser.parse_args()

    # Redirect all prints to stderr so only our JSON goes to stdout
    real_stdout = sys.stdout
    sys.stdout = sys.stderr

    env = make_env(args)

    # Signal ready
    real_stdout.write("READY\n")
    real_stdout.flush()

    # Restore stdout for JSON communication
    sys.stdout = real_stdout

    # Use readline() instead of `for line in sys.stdin` to avoid buffering issues
    while True:
        try:
            line = sys.stdin.readline()
        except Exception:
            break
        if not line:  # EOF
            break
        line = line.strip()
        if not line:
            continue

        try:
            cmd = json.loads(line)
        except json.JSONDecodeError:
            continue

        try:
            if cmd["cmd"] == "reset":
                session = cmd.get("session", 0)
                response = reset_env(env, args.wrapper, session)
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()

            elif cmd["cmd"] == "step":
                action = cmd["action"]
                response = step_env(env, args.wrapper, action)
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()

            elif cmd["cmd"] == "close":
                break
        except Exception as e:
            error_resp = {"error": str(e)}
            sys.stdout.write(json.dumps(error_resp) + "\n")
            sys.stdout.flush()

    env.close()


if __name__ == "__main__":
    main()
