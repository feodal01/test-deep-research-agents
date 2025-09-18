from __future__ import annotations

import argparse
import os
import subprocess
import sys


def run(cmd: list[str]) -> int:
    print("$", " ".join(cmd))
    return subprocess.call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description="Clone or update open_deep_research into src/agents/")
    parser.add_argument("--repo", default="https://github.com/langchain-ai/open_deep_research", help="Repo URL")
    parser.add_argument("--dest", default="src/agents/open_deep_research", help="Destination directory (relative to repo root)")
    parser.add_argument("--branch", default="main", help="Branch or tag to checkout")
    parser.add_argument("--depth", type=int, default=1, help="Git clone depth")
    args = parser.parse_args()

    # Resolve destination relative to repo root (three levels up from this file: scripts -> agents -> src -> repo)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
    dest = args.dest if os.path.isabs(args.dest) else os.path.join(repo_root, args.dest)
    print(f"Destination resolved to: {dest}")

    parent = os.path.dirname(dest)
    os.makedirs(parent, exist_ok=True)

    if not os.path.isdir(os.path.join(dest, ".git")):
        code = run(["git", "clone", "--depth", str(args.depth), "--branch", args.branch, args.repo, dest])
        if code != 0:
            print("Clone failed", file=sys.stderr)
            return code
    else:
        code = run(["git", "-C", dest, "fetch", "--all", "--tags", "--prune"])
        if code != 0:
            return code
        code = run(["git", "-C", dest, "checkout", args.branch])
        if code != 0:
            return code
        code = run(["git", "-C", dest, "pull", "--ff-only"])
        if code != 0:
            return code

    print(f"Ready at {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


