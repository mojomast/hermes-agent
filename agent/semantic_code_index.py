"""Project-local semantic code index for IDE-grade Hermes coding mode.

Bounded contexts:
- Semantic Code Index: cached model of symbols, definitions, references,
  diagnostics, and language/file metadata.
- Coding Primitive: tool-facing operation over the semantic index.

First version uses Python AST and lightweight JS/TS structure extraction. It is
not a grep wrapper: symbol definitions and Python references come from syntax
nodes; diagnostics use parsers/compilers where available.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from hermes_constants import get_hermes_home

CACHE_VERSION = "semantic_code_index.v2"
SUPPORTED_SUFFIXES = {".py", ".js", ".jsx", ".ts", ".tsx"}
SKIP_DIRS = {".git", "node_modules", "venv", ".venv", "__pycache__", "dist", "build", ".mypy_cache", ".pytest_cache"}
JS_DEF_RE = re.compile(r"(?:export\s+)?(?:(?:async\s+)?(?P<decl_kind>function|class)\s+(?P<decl_name>[A-Za-z_$][\w$]*)|(?P<var_kind>const|let|var)\s+(?P<var_name>[A-Za-z_$][\w$]*)\s*=|interface\s+(?P<interface_name>[A-Za-z_$][\w$]*)|type\s+(?P<type_name>[A-Za-z_$][\w$]*)\s*=)")
IDENT_RE = re.compile(r"\b[A-Za-z_$][\w$]*\b")


@dataclass(frozen=True)
class CodeSymbol:
    name: str
    kind: str
    language: str
    path: str
    line: int
    column: int
    end_line: Optional[int] = None
    parent: Optional[str] = None


@dataclass(frozen=True)
class CodeReference:
    name: str
    path: str
    line: int
    column: int
    context: str
    language: str


@dataclass(frozen=True)
class CodeDiagnostic:
    path: str
    line: int
    column: int
    severity: str
    message: str
    source: str


@dataclass
class SemanticCodeIndex:
    root: str
    version: str
    built_at: float
    file_count: int
    languages: Dict[str, int]
    fingerprint: str
    symbols: List[CodeSymbol]
    references: List[CodeReference]
    diagnostics: List[CodeDiagnostic]
    index_latency_ms: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def language_for(path: Path) -> str:
    if path.suffix == ".py":
        return "python"
    if path.suffix in {".ts", ".tsx"}:
        return "typescript"
    if path.suffix in {".js", ".jsx"}:
        return "javascript"
    return "unknown"


def iter_source_files(root: Path, max_files: int = 2_000) -> Iterable[Path]:
    count = 0
    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for name in files:
            path = Path(current) / name
            if path.suffix in SUPPORTED_SUFFIXES:
                yield path
                count += 1
                if count >= max_files:
                    return


def project_fingerprint(files: List[Path], root: Path) -> str:
    h = hashlib.sha256()
    for p in files:
        try:
            st = p.stat()
            h.update(str(p.relative_to(root)).encode())
            h.update(str(int(st.st_mtime_ns)).encode())
            h.update(str(st.st_size).encode())
        except OSError:
            continue
    return h.hexdigest()


def _line_context(lines: List[str], line: int) -> str:
    if 1 <= line <= len(lines):
        return lines[line - 1].strip()[:240]
    return ""


class _PythonVisitor(ast.NodeVisitor):
    def __init__(self, rel_path: str, lines: List[str], include_context: bool = False):
        self.rel_path = rel_path
        self.lines = lines
        self.include_context = include_context
        self.symbols: List[CodeSymbol] = []
        self.refs: List[CodeReference] = []
        self.stack: List[str] = []

    def _context_for(self, line: int) -> str:
        return _line_context(self.lines, line) if self.include_context else ""

    def _add_symbol(self, node: ast.AST, name: str, kind: str) -> None:
        line = getattr(node, "lineno", 1)
        col = getattr(node, "col_offset", 0)
        self.symbols.append(CodeSymbol(name=name, kind=kind, language="python", path=self.rel_path, line=line, column=col, end_line=getattr(node, "end_lineno", None), parent=(self.stack[-1] if self.stack else None)))
        # Treat definitions as semantic references too, matching LSP-style
        # reference lists that can include declaration/definition sites.
        self.refs.append(CodeReference(name=name, path=self.rel_path, line=line, column=col, context=self._context_for(line), language="python"))

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self._add_symbol(node, node.name, "class")
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._add_symbol(node, node.name, "function")
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(self, node: ast.Assign) -> Any:
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._add_symbol(target, target.id, "variable")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> Any:
        self.refs.append(CodeReference(name=node.id, path=self.rel_path, line=node.lineno, column=node.col_offset, context=self._context_for(node.lineno), language="python"))


def parse_python(path: Path, root: Path, include_context: bool = False) -> tuple[List[CodeSymbol], List[CodeReference], List[CodeDiagnostic]]:
    rel = str(path.relative_to(root))
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    try:
        tree = ast.parse(text, filename=rel)
    except SyntaxError as exc:
        return [], [], [CodeDiagnostic(path=rel, line=exc.lineno or 1, column=exc.offset or 0, severity="error", message=exc.msg, source="python.ast")]
    visitor = _PythonVisitor(rel, lines, include_context=include_context)
    visitor.visit(tree)
    return visitor.symbols, visitor.refs, []


def _js_var_decl_kind(line: str, match: re.Match[str]) -> str:
    initializer = line[match.end():].lstrip()
    if initializer.startswith("async "):
        initializer = initializer[6:].lstrip()
    if initializer.startswith("function") or "=>" in initializer:
        return "function"
    return "variable"


def parse_js_like(path: Path, root: Path, include_context: bool = False) -> tuple[List[CodeSymbol], List[CodeReference], List[CodeDiagnostic]]:
    rel = str(path.relative_to(root))
    text = path.read_text(encoding="utf-8", errors="replace")
    lang = language_for(path)
    symbols: List[CodeSymbol] = []
    refs: List[CodeReference] = []
    lines = text.splitlines()
    for line_no, line in enumerate(lines, start=1):
        for match in JS_DEF_RE.finditer(line):
            if match.group("decl_name"):
                name = match.group("decl_name")
                kind = match.group("decl_kind") or "function"
            elif match.group("interface_name"):
                name = match.group("interface_name")
                kind = "interface"
            elif match.group("type_name"):
                name = match.group("type_name")
                kind = "type"
            elif match.group("var_name"):
                name = match.group("var_name")
                kind = _js_var_decl_kind(line, match)
            else:
                continue
            symbols.append(CodeSymbol(name=name, kind=kind, language=lang, path=rel, line=line_no, column=match.start(), end_line=line_no))
        for match in IDENT_RE.finditer(line):
            refs.append(CodeReference(name=match.group(0), path=rel, line=line_no, column=match.start(), context=(line.strip()[:240] if include_context else ""), language=lang))
    diagnostics: List[CodeDiagnostic] = []
    if path.suffix == ".js":
        try:
            proc = subprocess.run(["node", "--check", str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
            if proc.returncode != 0:
                diagnostics.append(CodeDiagnostic(path=rel, line=1, column=0, severity="error", message=(proc.stderr or proc.stdout).strip()[:500], source="node --check"))
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return symbols, refs, diagnostics


def build_index(project_path: str | Path, max_files: int = 2_000, include_context: bool = False) -> SemanticCodeIndex:
    root = Path(project_path).expanduser().resolve()
    start = time.perf_counter()
    files = list(iter_source_files(root, max_files=max_files))
    fingerprint = project_fingerprint(files, root)
    languages: Dict[str, int] = {}
    symbols: List[CodeSymbol] = []
    refs: List[CodeReference] = []
    diagnostics: List[CodeDiagnostic] = []
    for path in files:
        lang = language_for(path)
        languages[lang] = languages.get(lang, 0) + 1
        try:
            if path.suffix == ".py":
                s, r, d = parse_python(path, root, include_context=include_context)
            else:
                s, r, d = parse_js_like(path, root, include_context=include_context)
            symbols.extend(s); refs.extend(r); diagnostics.extend(d)
        except OSError as exc:
            diagnostics.append(CodeDiagnostic(path=str(path), line=1, column=0, severity="error", message=str(exc), source="indexer"))
    latency = int((time.perf_counter() - start) * 1000)
    return SemanticCodeIndex(str(root), CACHE_VERSION, time.time(), len(files), languages, fingerprint, symbols, refs, diagnostics, latency)


def cache_path_for(project_path: str | Path) -> Path:
    root = Path(project_path).expanduser().resolve()
    digest = hashlib.sha256(str(root).encode()).hexdigest()[:16]
    return get_hermes_home() / "code_index_cache" / f"{digest}.json"


def load_or_build_index(project_path: str | Path, use_cache: bool = True, include_context: bool = False) -> SemanticCodeIndex:
    root = Path(project_path).expanduser().resolve()
    files = list(iter_source_files(root))
    fp = project_fingerprint(files, root)
    cp = cache_path_for(root)
    # Raw code contexts are intentionally never read from or written to the durable
    # cache. Opt-in context lookups rebuild in memory for this request only while
    # also replacing any legacy/raw cache file with a context-free v2 cache.
    if include_context:
        if use_cache:
            try:
                safe_idx = build_index(root, include_context=False)
                cp.parent.mkdir(parents=True, exist_ok=True)
                cp.write_text(json.dumps(safe_idx.to_dict(), sort_keys=True), encoding="utf-8")
            except Exception:
                pass
        return build_index(root, include_context=True)
    if use_cache and cp.exists():
        try:
            data = json.loads(cp.read_text(encoding="utf-8"))
            if data.get("fingerprint") == fp and data.get("version") == CACHE_VERSION:
                return SemanticCodeIndex(
                    root=data["root"], version=data["version"], built_at=data["built_at"], file_count=data["file_count"], languages=data["languages"], fingerprint=data["fingerprint"],
                    symbols=[CodeSymbol(**s) for s in data.get("symbols", [])], references=[CodeReference(**{**r, "context": ""}) for r in data.get("references", [])], diagnostics=[CodeDiagnostic(**d) for d in data.get("diagnostics", [])], index_latency_ms=0,
                )
        except Exception:
            pass
    idx = build_index(root, include_context=False)
    if use_cache:
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(idx.to_dict(), sort_keys=True), encoding="utf-8")
    return idx


def find_symbols(project_path: str | Path, query: str = "", kind: str | None = None, limit: int = 50) -> Dict[str, Any]:
    idx = load_or_build_index(project_path)
    q = (query or "").lower()
    hits = [s for s in idx.symbols if (not q or q in s.name.lower()) and (not kind or s.kind == kind)]
    return {"index": {"root": idx.root, "files": idx.file_count, "languages": idx.languages, "latency_ms": idx.index_latency_ms}, "symbols": [asdict(s) for s in hits[:limit]], "total": len(hits)}


def go_to_definition(project_path: str | Path, name: str) -> Dict[str, Any]:
    idx = load_or_build_index(project_path)
    hits = [s for s in idx.symbols if s.name == name]
    return {"definitions": [asdict(s) for s in hits], "total": len(hits)}


def list_references(project_path: str | Path, name: str, limit: int = 100, include_context: bool = False) -> Dict[str, Any]:
    idx = load_or_build_index(project_path, include_context=include_context)
    hits = [r for r in idx.references if r.name == name]
    return {"references": [asdict(r) for r in hits[:limit]], "total": len(hits)}


def get_diagnostics(project_path: str | Path, limit: int = 100) -> Dict[str, Any]:
    idx = load_or_build_index(project_path, use_cache=False)
    return {"diagnostics": [asdict(d) for d in idx.diagnostics[:limit]], "total": len(idx.diagnostics), "index": {"files": idx.file_count, "languages": idx.languages, "latency_ms": idx.index_latency_ms}}


def semantic_lookup(project_path: str | Path, operation: str, query: str = "", kind: str | None = None, limit: int = 50, include_context: bool = False) -> Dict[str, Any]:
    if operation in {"index", "summary"}:
        idx = load_or_build_index(project_path)
        return {"root": idx.root, "file_count": idx.file_count, "languages": idx.languages, "symbol_count": len(idx.symbols), "reference_count": len(idx.references), "diagnostic_count": len(idx.diagnostics), "index_latency_ms": idx.index_latency_ms, "fingerprint": idx.fingerprint[:16]}
    if operation == "find_symbol":
        return find_symbols(project_path, query=query, kind=kind, limit=limit)
    if operation == "definition":
        return go_to_definition(project_path, query)
    if operation == "references":
        return list_references(project_path, query, limit=limit, include_context=include_context)
    if operation == "diagnostics":
        return get_diagnostics(project_path, limit=limit)
    raise ValueError(f"unsupported semantic operation: {operation}")
