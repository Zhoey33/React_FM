"""Bridge script for WebShop — runs in conda webshop env, communicates via stdin/stdout JSON."""

import sys
import os
import json
import argparse
import warnings

# Ensure Java is available
os.environ["JAVA_HOME"] = "/opt/homebrew/Cellar/openjdk/25.0.2/libexec/openjdk.jdk/Contents/Home"
warnings.filterwarnings("ignore")

# Add WebShop to path
sys.path.insert(0, "/Users/zhoey/WebShop")

from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-products", type=int, default=1000)
    parser.add_argument("--observation-mode", type=str, default="text")
    args = parser.parse_args()

    # Redirect all prints to stderr so only our JSON goes to stdout
    real_stdout = sys.stdout
    sys.stdout = sys.stderr

    env = WebAgentTextEnv(
        observation_mode=args.observation_mode,
        num_products=args.num_products,
    )

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
                obs = env.reset(session)
                # obs is (text, None) tuple from WebAgentTextEnv
                obs_text = obs[0] if isinstance(obs, tuple) else str(obs)
                actions = env.get_available_actions()
                response = {
                    "observation": obs_text,
                    "actions": actions,
                }
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()

            elif cmd["cmd"] == "step":
                action = cmd["action"]
                obs, reward, done, info = env.step(action)
                actions = env.get_available_actions()
                response = {
                    "observation": obs,
                    "reward": reward,
                    "done": done,
                    "actions": actions,
                }
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
