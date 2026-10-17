"""Generated bug-fix tasks for the coding-agent predictive-execution benchmark (Track B).

Each task is a tiny package `pkg/` with 3 functions (from TEMPLATES) and a pytest file.
Exactly one function carries an injected bug (one of the template's MUTATIONS). A task
id = (template, mutation, companion seed). 10 templates x 3 mutations = 30 bug variants;
each variant appears with two different companion-function sets: one for TRAIN
(history for predictors), one for EVAL. Eval tasks are therefore distinct instances.
"""
from __future__ import annotations

import os
import random
import textwrap

TEMPLATES = {
    "clamp": ("def clamp(x, lo, hi):\n    return max(lo, min(x, hi))\n",
              ["assert clamp(5, 0, 3) == 3", "assert clamp(-1, 0, 3) == 0", "assert clamp(2, 0, 3) == 2"],
              [("min(x, hi)", "min(x, lo)"), ("max(lo,", "min(lo,"), ("return max(lo, min(x, hi))", "return x")]),
    "mean": ("def mean(xs):\n    if not xs:\n        return 0.0\n    return sum(xs) / len(xs)\n",
             ["assert mean([]) == 0.0", "assert mean([1, 2, 3]) == 2.0", "assert mean([4]) == 4.0"],
             [("sum(xs) / len(xs)", "sum(xs) / (len(xs) + 1)"), ("if not xs:", "if xs is None:"),
              ("return 0.0", "return None")]),
    "count_vowels": ("def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')\n",
                     ["assert count_vowels('Hello') == 2", "assert count_vowels('xyz') == 0",
                      "assert count_vowels('AEIOU') == 5"],
                     [("s.lower()", "s"), ("'aeiou'", "'aeio'"), ("sum(1", "sum(2")]),
    "fizzbuzz": ("def fizzbuzz(n):\n    if n % 15 == 0:\n        return 'FizzBuzz'\n    if n % 3 == 0:\n"
                 "        return 'Fizz'\n    if n % 5 == 0:\n        return 'Buzz'\n    return str(n)\n",
                 ["assert fizzbuzz(15) == 'FizzBuzz'", "assert fizzbuzz(9) == 'Fizz'", "assert fizzbuzz(10) == 'Buzz'",
                  "assert fizzbuzz(7) == '7'"],
                 [("n % 15 == 0", "n % 5 == 0"), ("return str(n)", "return n"), ("n % 3 == 0", "n % 4 == 0")]),
    "is_palindrome": ("def is_palindrome(s):\n    t = [c.lower() for c in s if c.isalnum()]\n    return t == t[::-1]\n",
                      ["assert is_palindrome('A man, a plan, a canal: Panama')", "assert not is_palindrome('abc')",
                       "assert is_palindrome('')"],
                      [("c.lower()", "c"), ("if c.isalnum()", ""), ("t == t[::-1]", "t == t")]),
    "running_max": ("def running_max(xs):\n    out, m = [], None\n    for x in xs:\n        m = x if m is None else max(m, x)\n"
                    "        out.append(m)\n    return out\n",
                    ["assert running_max([1, 3, 2, 5]) == [1, 3, 3, 5]", "assert running_max([]) == []",
                     "assert running_max([-2, -5]) == [-2, -2]"],
                    [("max(m, x)", "min(m, x)"), ("out.append(m)", "out.append(x)"), ("m is None", "m == 0")]),
    "dedupe": ("def dedupe(xs):\n    seen, out = set(), []\n    for x in xs:\n        if x not in seen:\n"
               "            seen.add(x)\n            out.append(x)\n    return out\n",
               ["assert dedupe([1, 2, 1, 3, 2]) == [1, 2, 3]", "assert dedupe([]) == []", "assert dedupe([4, 4]) == [4]"],
               [("if x not in seen:", "if x in seen:"), ("seen.add(x)", "pass"), ("return out", "return sorted(set(xs), reverse=True)")]),
    "word_freq": ("def word_freq(text):\n    d = {}\n    for w in text.lower().split():\n        d[w] = d.get(w, 0) + 1\n    return d\n",
                  ["assert word_freq('a b a') == {'a': 2, 'b': 1}", "assert word_freq('') == {}",
                   "assert word_freq('A a') == {'a': 2}"],
                  [("text.lower().split()", "text.split()"), ("d.get(w, 0) + 1", "1"), ("d.get(w, 0)", "d.get(w, 1)")]),
    "binary_search": ("def binary_search(xs, x):\n    lo, hi = 0, len(xs) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n"
                      "        if xs[mid] == x:\n            return mid\n        if xs[mid] < x:\n            lo = mid + 1\n"
                      "        else:\n            hi = mid - 1\n    return -1\n",
                      ["assert binary_search([1, 3, 5, 7], 5) == 2", "assert binary_search([1, 3, 5, 7], 4) == -1",
                       "assert binary_search([], 1) == -1", "assert binary_search([1, 3, 5, 7], 7) == 3"],
                      [("lo <= hi", "lo < hi"), ("lo = mid + 1", "lo = mid"), ("return -1", "return None")]),
    "flatten": ("def flatten(lst):\n    out = []\n    for x in lst:\n        if isinstance(x, list):\n            out.extend(x)\n"
                "        else:\n            out.append(x)\n    return out\n",
                ["assert flatten([1, [2, 3], 4]) == [1, 2, 3, 4]", "assert flatten([]) == []", "assert flatten([[1], [2]]) == [1, 2]"],
                [("out.extend(x)", "out.append(x)"), ("isinstance(x, list)", "isinstance(x, tuple)"), ("out.append(x)\n    return", "pass\n    return")]),
}


def task_ids():
    return [(t, m) for t in TEMPLATES for m in range(3)]


def make_task(root: str, template: str, mutation: int, split: str) -> dict:
    """Write the task workspace under root; returns metadata."""
    seed = sum(map(ord, template)) * 31 + mutation * 7 + (0 if split == "train" else 1000)
    rng = random.Random(seed)
    others = rng.sample([k for k in TEMPLATES if k != template], 2)
    os.makedirs(os.path.join(root, "pkg"), exist_ok=True)
    os.makedirs(os.path.join(root, "tests"), exist_ok=True)
    src = []
    tests = ["from pkg.core import *\n"]
    for name in [template] + others:
        code, ts, muts = TEMPLATES[name]
        if name == template:
            a, b = muts[mutation]
            assert a in code, (template, mutation)
            code = code.replace(a, b, 1)
        src.append(code)
        tests.append(f"\n\ndef test_{name}():\n" + textwrap.indent("\n".join(ts), "    ") + "\n")
    rng.shuffle(src)
    open(os.path.join(root, "pkg", "__init__.py"), "w").write("")
    open(os.path.join(root, "pkg", "core.py"), "w").write("\n\n".join(src))
    open(os.path.join(root, "tests", "test_core.py"), "w").write("".join(tests))
    return {"task": f"{template}#{mutation}#{split}", "family": template, "bug": f"{template}#{mutation}",
            "functions": [template] + others}
