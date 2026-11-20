#!/usr/bin/env python3
"""Point at code that may break a SOLID rule. Advisory: every line is a
question to answer by reading the code, not a verdict.

    python .claude/skills/solid-python/solid_check.py [paths...]

Standard library only. Default paths: agent_hailo.py cli_chat.py core
"""

import ast
import pathlib
import sys

CLASS_LINES, CLASS_METHODS, FUNCTION_LINES = 200, 15, 80   # S
BRANCHES = 4                                                # O
PARAMETERS = 7                                              # I
COLLABORATORS = 4                                           # D


def length(node):
    return node.end_lineno - node.lineno + 1


def type_switch(node):
    """Length of an if/elif chain that tests one name against constants or types, else 0."""
    subjects, count = set(), 0
    while isinstance(node, ast.If):
        test = node.test
        if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], (ast.Eq, ast.In)) \
                and isinstance(test.comparators[0], (ast.Constant, ast.Tuple, ast.Set)):
            subjects.add(ast.unparse(test.left))
        elif isinstance(test, ast.Call) and getattr(test.func, "id", "") == "isinstance" and test.args:
            subjects.add(ast.unparse(test.args[0]))
        else:
            return 0
        count += 1
        node = node.orelse[0] if len(node.orelse) == 1 else None
    return count if len(subjects) == 1 else 0


def check(path):
    tree = ast.parse(path.read_text())
    chained = {id(n.orelse[0]) for n in ast.walk(tree) if isinstance(n, ast.If) and len(n.orelse) == 1}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if length(node) > CLASS_LINES or len(methods) > CLASS_METHODS:
                yield node.lineno, length(node), "S", f"class {node.name}: {length(node)} lines, {len(methods)} methods. One reason to change?"
            for method in methods:
                if node.bases and any(isinstance(s, ast.Raise) and "NotImplementedError" in ast.unparse(s) for s in method.body):
                    yield method.lineno, 50, "L", f"{node.name}.{method.name} raises NotImplementedError in a subclass. Can it stand in for its base?"
                if method.name == "__init__":
                    made = sorted({c.func.id for c in ast.walk(method) if isinstance(c, ast.Call)
                                   and isinstance(c.func, ast.Name) and c.func.id[:1].isupper()
                                   and not c.func.id.endswith(("Error", "Exception", "Exit"))})
                    if len(made) >= COLLABORATORS:
                        yield method.lineno, 10 * len(made), "D", f"{node.name}.__init__ builds {len(made)} collaborators ({', '.join(made)}). Could they be passed in?"
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if length(node) > FUNCTION_LINES:
                yield node.lineno, length(node), "S", f"{node.name}(): {length(node)} lines. More than one job?"
            args = node.args
            count = len(args.posonlyargs) + len(args.args) + len(args.kwonlyargs) - (args.args[:1] and args.args[0].arg in ("self", "cls") and 1 or 0)
            if count > PARAMETERS:
                yield node.lineno, 10 * count, "I", f"{node.name}() takes {count} parameters. Does every caller need all of them, or is this one object?"
        elif isinstance(node, ast.If) and id(node) not in chained:
            n = type_switch(node)
            if n >= BRANCHES:
                yield node.lineno, 10 * n, "O", f"{n}-way branch on {ast.unparse(node.test)[:40]!r}. Does a new case mean editing this function?"


def main(argv):
    roots = [pathlib.Path(p) for p in (argv or ["agent_hailo.py", "cli_chat.py", "core"])]
    files = sorted(f for r in roots for f in ([r] if r.is_file() else r.rglob("*.py")) if "__pycache__" not in f.parts)
    findings = [(weight, f"{f}:{line}  [{rule}] {text}") for f in files for line, weight, rule, text in check(f)]
    for _, text in sorted(findings, reverse=True):
        print(text)
    print(f"{len(findings)} places to look at in {len(files)} files")


if __name__ == "__main__":
    main(sys.argv[1:])
