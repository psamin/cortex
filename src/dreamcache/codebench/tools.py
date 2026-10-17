"""Real tools for the coding-agent benchmark, with snapshot-addressed results.

Read-only/pure operations (run_tests, lint, deps_check) are functions of the workspace
snapshot (sha256 over all files) plus the tool fingerprint (python + tool versions), so
their results can be computed speculatively in an ISOLATED copy and reused only if the
snapshot hash and fingerprint match at request time. `edit` mutates the real workspace
and is never speculated.
"""
from __future__ import annotations

import hashlib
import re
import os
import shutil
import subprocess
import sys
import tempfile
import time

READ_ONLY = ("run_tests", "lint", "deps_check")


def snapshot_hash(root: str) -> str:
    h = hashlib.sha256()
    for dp, dn, fn in sorted(os.walk(root)):
        dn[:] = sorted(d for d in dn if not d.startswith((".", "__pycache__")))
        for f in sorted(fn):
            if f.endswith(".pyc"):
                continue
            p = os.path.join(dp, f)
            h.update(os.path.relpath(p, root).encode())
            h.update(open(p, "rb").read())
    return h.hexdigest()[:16]


def fingerprint() -> str:
    try:
        import ruff  # noqa: F401
        rv = subprocess.run([sys.executable, "-m", "ruff", "--version"], capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        rv = "ruff?"
    import pytest
    return f"py{sys.version_info[:3]}|pytest{pytest.__version__}|{rv}"


_WRAP = ("import resource, subprocess, sys; r = subprocess.run(sys.argv[1:]); "
         "u = resource.getrusage(resource.RUSAGE_CHILDREN); "
         "sys.stderr.write('__CPU__%f\\n' % (u.ru_utime + u.ru_stime)); sys.exit(r.returncode)")


def _run(cmd, cwd, timeout=60):
    """Run cmd; per-call CPU is measured inside a tiny wrapper process (RUSAGE_CHILDREN of
    the wrapper), so concurrent speculative runs do not contaminate each other's CPU time."""
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, "-c", _WRAP, *cmd], cwd=cwd, capture_output=True, text=True,
                       timeout=timeout, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    m = re.search(r"__CPU__([0-9.]+)", r.stderr)
    cpu = float(m.group(1)) if m else 0.0
    out = r.stdout + re.sub(r"__CPU__[0-9.]+\n?", "", r.stderr)
    # normalize run-dependent text so identical inputs give identical outputs
    out = re.sub(r"in \d+\.\d+s", "in <t>s", out).replace(os.path.abspath(cwd), "<ws>")[-1500:]
    return {"rc": r.returncode, "out": out, "seconds": time.perf_counter() - t0, "cpu": cpu}


def execute(op: str, root: str) -> dict:
    """Execute a read-only op against the workspace at `root` (no mutation)."""
    py = sys.executable
    if op == "run_tests":
        return _run([py, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--no-header", "tests"], root)
    if op == "lint":
        return _run([py, "-m", "ruff", "check", "--no-cache", "--select", "E,F", "--output-format", "concise", "pkg"], root)
    if op == "deps_check":
        return _run([py, "-c", "import importlib, pkg.core as m; importlib.reload(m); "
                     "print('imports ok:', sorted(n for n in dir(m) if not n.startswith('_')))"], root)
    raise ValueError(op)


def isolated_copy(root: str) -> str:
    d = tempfile.mkdtemp(prefix="cb-spec-")
    shutil.copytree(root, os.path.join(d, "ws"), ignore=shutil.ignore_patterns("__pycache__", ".*"))
    return os.path.join(d, "ws")


def read_file(root: str, rel: str) -> str:
    p = os.path.normpath(os.path.join(root, rel))
    if not p.startswith(os.path.abspath(root)) or not os.path.exists(p):
        return f"no such file: {rel}"
    return open(p).read()[-3000:]


def edit_function(root: str, func: str, code: str) -> str:
    """Replace the source of `def func(...)` in pkg/core.py with `code`."""
    p = os.path.join(root, "pkg", "core.py")
    src = open(p).read()
    blocks = src.split("\n\n")
    idx = [i for i, b in enumerate(blocks) if b.lstrip().startswith(f"def {func}(")]
    if not idx:
        return f"edit failed: function {func} not found"
    if f"def {func}(" not in code:
        return "edit failed: code must define the same function"
    blocks[idx[0]] = code.strip("\n") + "\n"
    open(p, "w").write("\n\n".join(blocks))
    return f"edited {func}"
