"""Strip # comments from every .py file under cka_ens/, scripts/, tests/.

Uses tokenize so it preserves docstrings, string literals (including
strings that contain '#'), and whitespace. Leaves shebangs intact.
Removes whole-line comments entirely (cleaning up the blank line);
strips trailing inline comments and any trailing whitespace they leave.
"""
from __future__ import annotations

import io
import sys
import tokenize
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
TARGETS = ["cka_ens", "scripts", "tests"]
SKIP_FILES = {"scripts/strip_comments.py"}


def strip_file(src_text: str) -> str:
    """Return src_text with # comments removed. Preserves strings/docstrings."""
    out_tokens = []
    g = tokenize.generate_tokens(io.StringIO(src_text).readline)
    for tok in g:
        if tok.type == tokenize.COMMENT:
            continue
        out_tokens.append(tok)
    rebuilt = tokenize.untokenize(out_tokens)

    cleaned_lines = []
    for ln in rebuilt.splitlines():
        stripped_right = ln.rstrip()
        cleaned_lines.append(stripped_right)

    text = "\n".join(cleaned_lines)
    while "\n\n\n\n" in text:
        text = text.replace("\n\n\n\n", "\n\n\n")
    if not text.endswith("\n"):
        text += "\n"
    return text


def iter_py(targets: Iterable[str]):
    for t in targets:
        base = ROOT / t
        if not base.exists():
            continue
        for p in base.rglob("*.py"):
            rel = str(p.relative_to(ROOT))
            if rel in SKIP_FILES:
                continue
            yield p


def main() -> int:
    n_changed = 0
    for p in iter_py(TARGETS):
        src = p.read_text()
        if src.startswith("#!"):
            shebang, rest = src.split("\n", 1)
            new = shebang + "\n" + strip_file(rest)
        else:
            new = strip_file(src)
        if new != src:
            p.write_text(new)
            n_changed += 1
            print(f"stripped: {p.relative_to(ROOT)}")
    print(f"\n{n_changed} file(s) modified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
