"""Static scan of a proposed strategy.py. Runs BEFORE any code executes.

Rejects (ScanError with a reason) on:
  * syntax errors, files over SCAN_MAX_LINES, missing `def strategy(df)`
  * imports outside {numpy, pandas, math}
  * any hardcoded date / year / datetime literal or datetime-ish name
  * any file / network / env / process / introspection access
  * dunder attribute access
  * free variable names outside the function's arguments / locals /
    module-level constants & helpers / a small builtin whitelist
  * module-level code other than imports, literal constants, functions, docstring
  * rolling(..., center=True) anywhere
  * shift/diff/pct_change with a non-positive-literal period
  * forward-reaching slices on iloc/loc/values ([k:] or [:-k])
  * np.roll, bfill/backfill, interpolate, convolve, resample and friends
"""
from __future__ import annotations

import ast
import re

from harness import config


class ScanError(ValueError):
    pass


FORBIDDEN_NAMES = {
    # I/O, network, env, process
    "open", "read_csv", "read_parquet", "read_pickle", "read_json", "read_excel", "read_hdf",
    "read_sql", "read_feather", "read_table", "read_fwf", "read_clipboard", "to_csv",
    "to_parquet", "to_pickle", "to_json", "to_hdf", "requests", "urllib", "urllib3", "httpx",
    "socket", "http", "ftplib", "os", "environ", "getenv", "putenv", "sys", "pathlib",
    "Path", "subprocess", "shutil", "glob", "io", "pickle", "marshal", "shelve", "sqlite3",
    "tempfile", "ctypes", "multiprocessing", "threading", "signal", "importlib", "builtins",
    "webbrowser", "input", "print",
    # dynamic execution / introspection
    "eval", "exec", "compile", "__import__", "globals", "locals", "vars", "getattr",
    "setattr", "delattr", "breakpoint", "type", "object", "super", "memoryview",
    # time / calendar
    "datetime", "date", "time", "Timestamp", "to_datetime", "DatetimeIndex", "date_range",
    "bdate_range", "strftime", "strptime", "Timedelta", "to_timedelta", "Period",
    "PeriodIndex", "calendar", "zoneinfo", "pytz", "dateutil", "timestamp", "fromtimestamp",
    # forward-looking / non-causal helpers
    "roll", "bfill", "backfill", "interpolate", "convolve", "correlate", "fft", "resample",
    "asfreq", "reindex", "sort_values", "sort_index", "rank", "qcut", "cut",
}
BUILTIN_WHITELIST = {
    "len", "range", "abs", "min", "max", "float", "int", "bool", "round", "enumerate",
    "zip", "isinstance", "list", "tuple", "dict", "set", "True", "False", "None",
    "sorted", "reversed", "map", "filter", "any", "all", "pow", "divmod", "str",
    "ValueError", "TypeError", "ZeroDivisionError", "Exception",
}
SHIFT_LIKE = {"shift", "diff", "pct_change"}
DATE_RE = re.compile(r"\b(19|20)\d{2}[-/.](0?[1-9]|1[0-2])([-/.](0?[1-9]|[12]\d|3[01]))?\b")
SLICE_ATTRS = {"iloc", "loc", "values", "to_numpy", "array"}


def _const_int(node: ast.AST, consts: dict[str, object]) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return int(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _const_int(node.operand, consts)
        return None if v is None else -v
    if isinstance(node, ast.Name) and isinstance(consts.get(node.id), (int, float)):
        return int(consts[node.id])
    return None


def _literal_ok(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float, str, bool)) or node.value is None
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(_literal_ok(e) for e in node.elts)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _literal_ok(node.operand)
    if isinstance(node, ast.BinOp):
        return _literal_ok(node.left) and _literal_ok(node.right)
    return False


def _year_like(v) -> bool:
    lo, hi = config.SCAN_YEAR_RANGE
    return isinstance(v, int) and not isinstance(v, bool) and lo <= v <= hi


class _Scanner(ast.NodeVisitor):
    def __init__(self, module_names: set[str], consts: dict[str, object]):
        self.module_names = module_names
        self.consts = consts
        self.errors: list[str] = []
        self.scopes: list[set[str]] = []

    def err(self, node, msg):
        self.errors.append(f"line {getattr(node, 'lineno', '?')}: {msg}")

    # ---- names / attributes -------------------------------------------
    def visit_Name(self, node: ast.Name):
        if node.id in FORBIDDEN_NAMES:
            self.err(node, f"forbidden name `{node.id}`")
        elif isinstance(node.ctx, ast.Load) and self.scopes:
            if node.id not in self.scopes[-1] and node.id not in self.module_names \
                    and node.id not in BUILTIN_WHITELIST:
                self.err(node, f"free variable `{node.id}` (not an argument, local, import or module constant)")
        elif isinstance(node.ctx, ast.Store) and self.scopes:
            self.scopes[-1].add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if node.attr in FORBIDDEN_NAMES:
            self.err(node, f"forbidden attribute `.{node.attr}`")
        if node.attr.startswith("__") and node.attr.endswith("__"):
            self.err(node, f"dunder attribute `.{node.attr}`")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant):
        if _year_like(node.value):
            self.err(node, f"year-like integer literal {node.value}")
        if isinstance(node.value, str) and DATE_RE.search(node.value):
            self.err(node, f"date-like string literal {node.value!r}")

    # ---- calls -------------------------------------------------------
    def visit_Call(self, node: ast.Call):
        for kw in node.keywords:
            if kw.arg == "center" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                self.err(node, "center=True is non-causal")
            if kw.arg == "method" and isinstance(kw.value, ast.Constant) and kw.value.value in ("bfill", "backfill"):
                self.err(node, "backward fill is non-causal")
        fname = node.func.attr if isinstance(node.func, ast.Attribute) else (node.func.id if isinstance(node.func, ast.Name) else None)
        if fname in SHIFT_LIKE:
            period_node = None
            if node.args:
                period_node = node.args[0]
            for kw in node.keywords:
                if kw.arg == "periods":
                    period_node = kw.value
            if period_node is not None:
                v = _const_int(period_node, self.consts)
                if v is None:
                    self.err(node, f"{fname}() period must be a positive integer literal or module constant")
                elif v < 0 or (fname == "shift" and v == 0):
                    self.err(node, f"{fname}({v}) reaches forward in time")
        self.generic_visit(node)

    # ---- slices ------------------------------------------------------
    def visit_Subscript(self, node: ast.Subscript):
        base_attr = node.value.attr if isinstance(node.value, ast.Attribute) else None
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
            base_attr = node.value.func.attr
        if isinstance(node.slice, ast.Slice) and (base_attr in SLICE_ATTRS or isinstance(node.value, ast.Name)):
            lo = _const_int(node.slice.lower, self.consts) if node.slice.lower is not None else None
            hi = _const_int(node.slice.upper, self.consts) if node.slice.upper is not None else None
            if (lo is not None and lo > 0) or (hi is not None and hi < 0):
                self.err(node, "offset slice ([k:] / [:-k]) misaligns bars — use .shift(k) instead")
        if isinstance(node.slice, ast.UnaryOp) and isinstance(node.slice.op, ast.USub):
            self.err(node, "negative index reaches the end of the series (future)")
        self.generic_visit(node)

    # ---- scopes ------------------------------------------------------
    def _visit_func(self, node):
        args = node.args
        names = {a.arg for a in args.args + args.posonlyargs + args.kwonlyargs}
        if args.vararg:
            names.add(args.vararg.arg)
        if args.kwarg:
            names.add(args.kwarg.arg)
        self.scopes.append(names)
        for stmt in node.body:
            self.visit(stmt)
        self.scopes.pop()

    visit_FunctionDef = _visit_func
    visit_AsyncFunctionDef = _visit_func

    def visit_Lambda(self, node: ast.Lambda):
        self.scopes.append(set(self.scopes[-1]) if self.scopes else set())
        self.scopes[-1].update(a.arg for a in node.args.args)
        self.visit(node.body)
        self.scopes.pop()

    def _visit_comp(self, node):
        if self.scopes:
            self.scopes.append(set(self.scopes[-1]))
        else:
            self.scopes.append(set())
        for gen in node.generators:
            self.visit(gen.iter)
            self.visit(gen.target)
            for cond in gen.ifs:
                self.visit(cond)
        if isinstance(node, ast.DictComp):
            self.visit(node.key)
            self.visit(node.value)
        else:
            self.visit(node.elt)
        self.scopes.pop()

    visit_ListComp = _visit_comp
    visit_SetComp = _visit_comp
    visit_GeneratorExp = _visit_comp
    visit_DictComp = _visit_comp

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            mod = alias.name.split(".")[0]
            if mod not in config.SCAN_ALLOWED_IMPORTS:
                self.err(node, f"import of `{mod}` not allowed (only {sorted(config.SCAN_ALLOWED_IMPORTS)})")
        if self.scopes:
            self.err(node, "imports inside functions are not allowed")

    def visit_ImportFrom(self, node: ast.ImportFrom):
        mod = (node.module or "").split(".")[0]
        if mod not in config.SCAN_ALLOWED_IMPORTS:
            self.err(node, f"import of `{mod}` not allowed (only {sorted(config.SCAN_ALLOWED_IMPORTS)})")
        for alias in node.names:
            if alias.name in FORBIDDEN_NAMES:
                self.err(node, f"forbidden name `{alias.name}`")
        if self.scopes:
            self.err(node, "imports inside functions are not allowed")

    def visit_Global(self, node):
        self.err(node, "global statement")

    def visit_Nonlocal(self, node):
        self.err(node, "nonlocal statement")

    def visit_ClassDef(self, node):
        self.err(node, "class definitions are not allowed in strategy.py")


def scan_source(source: str) -> None:
    """Raise ScanError with every reason found; return None if clean."""
    lines = source.splitlines()
    if len(lines) > config.SCAN_MAX_LINES:
        raise ScanError(f"{len(lines)} lines > SCAN_MAX_LINES={config.SCAN_MAX_LINES}")
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise ScanError(f"syntax error: {e.msg} (line {e.lineno})") from e

    errors: list[str] = []
    module_names: set[str] = set()
    consts: dict[str, object] = {}
    has_strategy = False

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            root = (node.module or "").split(".")[0] if isinstance(node, ast.ImportFrom) else None
            for alias in node.names:
                mod = root if root is not None else alias.name.split(".")[0]
                if mod not in config.SCAN_ALLOWED_IMPORTS:
                    errors.append(f"line {node.lineno}: import of `{mod}` not allowed (only {sorted(config.SCAN_ALLOWED_IMPORTS)})")
                module_names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.FunctionDef):
            module_names.add(node.name)
            if node.name == "strategy":
                has_strategy = True
                if len(node.args.args) != 1 or node.args.vararg or node.args.kwonlyargs or node.args.kwarg:
                    errors.append(f"line {node.lineno}: strategy must take exactly one argument (df)")
        elif isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets) and _literal_ok(node.value):
            for t in node.targets:
                module_names.add(t.id)
                try:
                    consts[t.id] = ast.literal_eval(node.value)
                except Exception:  # noqa: BLE001
                    pass
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            pass  # docstring
        else:
            errors.append(f"line {node.lineno}: module-level statement not allowed ({type(node).__name__}); only imports, literal constants, functions")

    if not has_strategy:
        errors.append("no `def strategy(df)` found")

    for line_no, line in enumerate(lines, 1):
        if DATE_RE.search(line):
            errors.append(f"line {line_no}: date-like literal in source")

    sc = _Scanner(module_names, consts)
    sc.visit(tree)
    errors.extend(sc.errors)

    if errors:
        seen, uniq = set(), []
        for e in errors:
            if e not in seen:
                seen.add(e)
                uniq.append(e)
        raise ScanError("; ".join(uniq[:12]))
