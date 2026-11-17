"""Launch the bundled Cosmos Policy training or Bench2Dex evaluation code.

Training defaults to the Cosmos command's read-only plan mode; execution needs
its explicit --execute flag.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

BENCH_REPO = Path(__file__).resolve().parents[2]
POLICY_REPO = Path(__file__).resolve().parent
PATH_FLAGS = {
    "--manifest", "--embedding", "--base-model", "--tokenizer", "--checkpoint",
    "--scene", "--anchor-dir", "--robot-map", "--output", "--embodiment-manifest",
}
OWNED_FLAGS = {"--repo", "--policy-repo", "--bench-repo", "--policy-python", "--sim-python"}


def _path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    # Keep a virtualenv's python symlink: resolving it selects the base
    # interpreter and loses that environment's installed packages.
    return (path if path.is_absolute() else base / path).absolute()


def load_config(path: Path, mode: str) -> dict[str, Path]:
    path = path.expanduser().resolve()
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("config must be a JSON object")
    required = ("cosmos_python",)
    if mode == "eval":
        required += ("sim_python",)
    if any(not isinstance(raw.get(key), str) or not raw[key].strip() for key in required):
        raise ValueError(f"config requires nonempty {', '.join(required)}")
    config = {key: _path(raw[key], path.parent) for key in required}
    if not (POLICY_REPO / "cosmos_policy/experiments/robot/bench2dex/train_embodiment.py").is_file():
        raise ValueError(f"bundled Cosmos training adapter missing from {POLICY_REPO}")
    if mode == "eval" and not (POLICY_REPO / "bin/bench2dex_final_eval.py").is_file():
        raise ValueError(f"bundled Cosmos evaluation runner missing from {POLICY_REPO}")
    for key in ("cosmos_python", "sim_python"):
        if key in config and not config[key].is_file():
            raise ValueError(f"{key} is not a file: {config[key]}")
    return config


def normalize_arguments(arguments: list[str], base: Path) -> list[str]:
    result = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        flag, equals, value = argument.partition("=")
        if flag in OWNED_FLAGS:
            raise ValueError(f"{flag} is set by bridge.py")
        if flag in PATH_FLAGS:
            if equals:
                result.append(f"{flag}={_path(value, base)}")
            else:
                if index + 1 >= len(arguments):
                    raise ValueError(f"{flag} needs a path")
                result.extend((flag, str(_path(arguments[index + 1], base))))
                index += 1
        else:
            result.append(argument)
        index += 1
    return result


def command(config: dict[str, Path], mode: str, arguments: list[str]) -> list[str]:
    python = config["cosmos_python"]
    if mode == "train":
        return [str(python), "-m", "cosmos_policy.experiments.robot.bench2dex.train_embodiment",
                "--repo", str(POLICY_REPO), *arguments]
    return [str(python), str(POLICY_REPO / "bin/bench2dex_final_eval.py"),
            "--policy-repo", str(POLICY_REPO), "--bench-repo", str(BENCH_REPO),
            "--policy-python", str(python), "--sim-python", str(config["sim_python"]),
            *arguments]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--print-command", action="store_true", help="Inspect the resolved command without running it")
    parser.add_argument("mode", choices=("train", "eval"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help="Arguments forwarded to the Cosmos runner")
    args = parser.parse_args()
    forwarded = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    try:
        config = load_config(args.config, args.mode)
        resolved = command(config, args.mode, normalize_arguments(forwarded, Path.cwd()))
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        parser.error(str(error))
    if args.print_command:
        print(json.dumps(resolved, indent=2))
        return 0
    return subprocess.call(resolved, cwd=POLICY_REPO)


if __name__ == "__main__":
    sys.exit(main())
