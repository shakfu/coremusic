"""Validate the code that appears in the documentation.

Two things are checked:

- Every `--8<--` include in a doc page resolves to a real file and, when a
  section is named, to a real section marker inside it.
- Every python block still written inline in a doc page parses, and every
  coremusic name it uses actually exists.

The second check is the cheap half of `test_examples.py`: it cannot tell
whether a snippet does what it claims, but it does catch the failure mode this
documentation had - blocks written against modules and attributes that were
renamed or removed.
"""

import ast
import importlib
import os
import re

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(REPO_ROOT, "docs")

# Design notes, not user documentation: they describe the code as it was when
# the note was written and are excluded from the site (`not_in_nav` in
# mkdocs.yml).
EXCLUDED_DIRS = {"dev"}

PYTHON_BLOCK = re.compile(r"^```python\n(.*?)^```", re.M | re.S)
INCLUDE = re.compile(r'^\s*--8<--\s+"([^"]+)"', re.M)


def find_docs():
    """Every published documentation page, as repo-relative paths."""
    found = [os.path.relpath(os.path.join(REPO_ROOT, "README.md"), REPO_ROOT)]
    for dirpath, dirnames, filenames in os.walk(DOCS_DIR):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]
        for filename in sorted(filenames):
            if filename.endswith(".md"):
                path = os.path.join(dirpath, filename)
                found.append(os.path.relpath(path, REPO_ROOT))
    return sorted(found)


DOCS = find_docs()


def python_blocks(path):
    """Yield (line number, source) for every inline python block."""
    text = open(os.path.join(REPO_ROOT, path)).read()
    for match in PYTHON_BLOCK.finditer(text):
        line = text[: match.start()].count("\n") + 1
        yield line, match.group(1)


def unresolved_names(source):
    """Report coremusic names used by `source` that do not exist.

    Only imports are resolved, plus attribute access on an imported module
    alias. That covers the way the documentation actually goes stale.
    """
    problems = []
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"does not parse: {e}"]

    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] != "coremusic":
                    continue
                try:
                    importlib.import_module(alias.name)
                except Exception as e:
                    problems.append(f"import {alias.name}: {e}")
                    continue
                aliases[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom):
            if not node.module or node.module.split(".")[0] != "coremusic":
                continue
            try:
                module = importlib.import_module(node.module)
            except Exception as e:
                problems.append(f"from {node.module}: {e}")
                continue
            for alias in node.names:
                if alias.name != "*" and not hasattr(module, alias.name):
                    problems.append(f"{node.module} has no {alias.name!r}")

    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if not isinstance(node.value, ast.Name) or node.value.id not in aliases:
            continue
        module_name = aliases[node.value.id]
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        if not hasattr(module, node.attr):
            problems.append(f"{module_name} has no {node.attr!r}")

    return sorted(set(problems))


@pytest.mark.parametrize("doc", DOCS)
def test_includes_resolve(doc):
    """Every `--8<--` include points at a file, and a section that exists."""
    text = open(os.path.join(REPO_ROOT, doc)).read()
    for target in INCLUDE.findall(text):
        path, _, section = target.partition(":")
        full = os.path.join(REPO_ROOT, path)
        assert os.path.exists(full), f"{doc}: include {path!r} does not exist"
        if section:
            marker = f"--8<-- [start:{section}]"
            assert marker in open(full).read(), (
                f"{doc}: {path} has no section {section!r}"
            )


@pytest.mark.parametrize("doc", DOCS)
def test_inline_snippets_use_real_names(doc):
    """Inline python blocks only reference names coremusic actually exports."""
    failures = []
    for line, source in python_blocks(doc):
        for problem in unresolved_names(source):
            failures.append(f"{doc}:{line}: {problem}")
    assert not failures, "\n".join(failures)
