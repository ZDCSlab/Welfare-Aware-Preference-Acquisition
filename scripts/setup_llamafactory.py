"""Fetch the supported LlamaFactory revision and apply the soft-label patch."""

import argparse
import subprocess
from pathlib import Path
from wapa.training.dpo import REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, help="External checkout location")
    args = parser.parse_args()
    root = Path(args.directory).resolve()
    patch = (
        Path(__file__).resolve().parents[1]
        / "patches/llamafactory-soft-label-dpo.patch"
    )

    def git(*parts):
        return subprocess.run(
            ["git", "-C", str(root), *parts], check=True, capture_output=True, text=True
        )

    if not root.exists() or not any(root.iterdir()):
        root.mkdir(parents=True, exist_ok=True)
        git("init")
        git("remote", "add", "origin", "https://github.com/hiyouga/LLaMA-Factory.git")
        git("fetch", "--depth", "1", "origin", REVISION)
        git("checkout", "--detach", "FETCH_HEAD")
    if git("rev-parse", "HEAD").stdout.strip() != REVISION:
        raise RuntimeError(
            "existing checkout has a different revision; use an empty directory"
        )
    applied = subprocess.run(
        ["git", "-C", str(root), "apply", "--reverse", "--check", str(patch)],
        capture_output=True,
    )
    if applied.returncode:
        git("apply", "--check", str(patch))
        git("apply", str(patch))
    print(f"Patched checkout ready: {root}")
    print(f'Install in your training environment: python -m pip install -e "{root}"')


if __name__ == "__main__":
    main()
