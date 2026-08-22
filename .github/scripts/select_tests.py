#!/usr/bin/env python3
"""Select pytest targets based on the files changed in a commit range.

The workflow uses this helper to keep CI focused on the relevant test files
for source changes, while still falling back to the full suite when the change
is broad or the mapping is ambiguous. Test-only changes are intentionally
skipped so they do not block pushes when no application code changed.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

APP_PREFIXES = ("chatbot/", "cli/", "eval/", "rag/")

# Known source-to-test mappings for this repository.
PACKAGE_TEST_DIRS = [
    ("rag/validation_layers/", Path("tests/rag/verification_layer")),
    ("rag/ingestion/", Path("tests/rag/ingestion")),
    ("rag/retrieval/", Path("tests/rag/retrieval")),
    ("rag/context_assembly/", Path("tests/rag/context_assembly")),
    ("rag/translation/", Path("tests/rag/translation")),
    ("chatbot/", Path("tests/chatbot")),
]


def run_git(args: list[str]) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def changed_files(base: str, head: str) -> list[Path]:
    zero_sha = "0" * 40
    if not base or base == zero_sha:
        lines = run_git(["show", "--pretty=", "--name-only", head])
    else:
        lines = run_git(["diff", "--name-only", base, head, "--"])
    return [Path(line) for line in lines]


def target_exists(target: Path) -> bool:
    return (ROOT / target).exists()


def expand_target(target: Path) -> set[Path]:
    """Expand a target path into concrete pytest file targets."""

    absolute = ROOT / target
    if absolute.is_file():
        return {target}

    if absolute.is_dir():
        return {
            Path(path.relative_to(ROOT).as_posix())
            for path in sorted(absolute.rglob("test_*.py"))
            if path.is_file()
        }

    return set()


def map_source_file(path: Path) -> Path | None:
    path_str = path.as_posix()

    if path.name == "__init__.py":
        return None

    if path_str == "chatbot/prompt.py":
        return Path("tests/chatbot/test_prompt.py")
    if path_str == "chatbot/workflow.py":
        return Path("tests/chatbot/test_workflow.py")
    if path_str in {"chatbot/utils.py", "chatbot/state.py"}:
        return Path("tests/chatbot")
    if path_str.startswith("cli/") and path.suffix == ".py":
        return Path("tests/cli")

    for prefix, test_dir in PACKAGE_TEST_DIRS:
        if path_str.startswith(prefix) and path.suffix == ".py":
            return test_dir / f"test_{path.stem}.py"

    if path_str.startswith("rag/") and path.suffix == ".py":
        return Path("tests/rag")

    if path_str.startswith("chatbot/") and path.suffix == ".py":
        return Path("tests/chatbot")

    return None


def select_targets(files: list[Path]) -> str:
    run_all = False
    targets: set[Path] = set()

    for path in files:
        path_str = path.as_posix()

        if path.name == "__init__.py":
            continue

        if path_str.startswith("tests/"):
            continue

        if not path_str.startswith(APP_PREFIXES):
            continue

        if path.suffix == ".py":
            mapped = map_source_file(path)
            if mapped is None:
                run_all = True
                continue

            resolved = expand_target(mapped)
            if not resolved:
                run_all = True
                continue

            targets.update(resolved)

    if run_all:
        return "ALL"

    ordered = sorted(targets, key=lambda p: p.as_posix())
    return "\n".join(target.as_posix() for target in ordered)


def main() -> int:
    parser = argparse.ArgumentParser(description="Select pytest targets for a commit range.")
    parser.add_argument("--base", required=True, help="Base commit SHA")
    parser.add_argument("--head", required=True, help="Head commit SHA")
    args = parser.parse_args()

    files = changed_files(args.base, args.head)
    print(select_targets(files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
