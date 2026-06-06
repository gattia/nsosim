"""Staleness gate for the docs site cross-references.

The narrative docs (``docs/*.md``) link to the API reference with mkdocstrings
autorefs of the form ``[`label`][nsosim.module.symbol]``. If a referenced
function is renamed or deleted, that link silently rots — exactly the failure
mode we want to catch.

This test parses every ``nsosim.<module>.<symbol>`` autoref target out of the
Markdown and asserts the symbol is actually defined in the corresponding source
file. It uses the standard-library ``ast`` module only: it does NOT import
nsosim (so it runs without opensim/torch/NSM) and it does NOT require mkdocs to
be installed — it works in the normal test environment.

If you rename a documented function, this test fails until the docs are updated.
"""

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs"

# mkdocstrings autoref target: ...][nsosim.a.b.c]
AUTOREF = re.compile(r"\]\[(nsosim\.[A-Za-z0-9_.]+)\]")


def _collect_targets():
    """Return {dotted_target: source_md_relpath} for every autoref in docs/."""
    targets = {}
    for md in sorted(DOCS_DIR.rglob("*.md")):
        text = md.read_text()
        for m in AUTOREF.finditer(text):
            targets.setdefault(m.group(1), md.relative_to(REPO_ROOT).as_posix())
    return targets


def _module_symbols(py_path: Path):
    """Top-level def/class names defined in a .py file (via ast, no import)."""
    tree = ast.parse(py_path.read_text(), filename=str(py_path))
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):  # module-level constants (e.g. OSIM_TO_NSM_TRANSFORM)
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
    return names


def _resolve(target: str):
    """Map ``nsosim.a.b.symbol`` -> (source_file, symbol). symbol is the last part."""
    parts = target.split(".")
    symbol = parts[-1]
    module_parts = parts[:-1]  # nsosim.a.b
    py_file = REPO_ROOT.joinpath(*module_parts).with_suffix(".py")
    return py_file, symbol


TARGETS = _collect_targets()


def test_docs_have_autoref_targets():
    """Guard against a regex/relayout change silently finding nothing to check."""
    assert TARGETS, "no nsosim.* autoref targets found in docs/ — did the link style change?"


@pytest.mark.parametrize("target", sorted(TARGETS))
def test_doc_reference_resolves(target):
    """Every nsosim.module.symbol referenced in the docs must exist in source."""
    py_file, symbol = _resolve(target)
    assert py_file.is_file(), (
        f"doc reference {target!r} (in {TARGETS[target]}) points at module file "
        f"{py_file.relative_to(REPO_ROOT)} which does not exist"
    )
    symbols = _module_symbols(py_file)
    assert symbol in symbols, (
        f"doc reference {target!r} (in {TARGETS[target]}) is stale: "
        f"{symbol!r} is not defined in {py_file.relative_to(REPO_ROOT)}"
    )
