"""One-shot tool: reduce every docstring in cka_ens/scripts/tests to its first non-empty line."""
from __future__ import annotations

import ast
import io
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGETS = ["cka_ens", "scripts", "tests"]
SKIP = {"scripts/compress_docstrings.py", "scripts/strip_comments.py"}


def first_line(text: str) -> str:
    for ln in text.splitlines():
        s = ln.strip()
        if s:
            return s
    return ""


def compress(src: str) -> str:
    """Return src with every module/class/function docstring reduced to one line."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src

    edits = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if not body:
                continue
            first = body[0]
            if not (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                continue
            text = first.value.value
            if "\n" not in text:
                continue
            edits.append((first.lineno, first.end_lineno, first.col_offset, first_line(text)))

    if not edits:
        return src

    edits.sort(reverse=True)
    lines = src.splitlines(keepends=True)
    for start, end, col, oneline in edits:
        indent = " " * col
        replacement = f'{indent}"""{oneline}"""' + "\n"
        new_lines = lines[: start - 1] + [replacement] + lines[end:]
        lines = new_lines
    return "".join(lines)


def iter_py():
    for t in TARGETS:
        base = ROOT / t
        if not base.exists():
            continue
        for p in base.rglob("*.py"):
            rel = str(p.relative_to(ROOT))
            if rel in SKIP:
                continue
            yield p


def main() -> int:
    n = 0
    for p in iter_py():
        src = p.read_text()
        new = compress(src)
        if new != src:
            p.write_text(new)
            n += 1
            print(f"compressed: {p.relative_to(ROOT)}")
    print(f"\n{n} file(s) modified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
