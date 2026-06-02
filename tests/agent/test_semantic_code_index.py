import json

from agent.semantic_code_index import cache_path_for, get_diagnostics, go_to_definition, list_references, semantic_lookup


def test_python_semantic_index_symbols_definitions_references_and_diagnostics(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "app.py").write_text(
        "class Greeter:\n"
        "    def hello(self, name):\n"
        "        return format_name(name)\n"
        "\n"
        "def format_name(value):\n"
        "    return value.title()\n",
        encoding="utf-8",
    )
    (project / "broken.py").write_text("def nope(:\n", encoding="utf-8")

    summary = semantic_lookup(project, "summary")
    assert summary["file_count"] == 2
    assert summary["languages"]["python"] == 2
    assert summary["symbol_count"] >= 3
    assert summary["diagnostic_count"] == 1
    assert summary["index_latency_ms"] >= 0

    defs = go_to_definition(project, "format_name")
    assert defs["definitions"][0]["path"] == "app.py"
    assert defs["definitions"][0]["line"] == 5

    refs = list_references(project, "format_name")
    assert refs["total"] >= 2  # definition + call site from AST Name refs
    assert any(r["line"] == 3 for r in refs["references"])
    assert all(r.get("context", "") == "" for r in refs["references"])

    refs_with_context = list_references(project, "format_name", include_context=True)
    assert any("return format_name(name)" in r.get("context", "") for r in refs_with_context["references"])

    diags = get_diagnostics(project)
    assert diags["total"] == 1
    assert diags["diagnostics"][0]["source"] == "python.ast"


def test_javascript_typescript_symbol_extraction(tmp_path):
    project = tmp_path / "web"
    project.mkdir()
    (project / "index.ts").write_text(
        "export interface User { name: string }\n"
        "export function loadUser(): User { return { name: 'Ada' } }\n"
        "const helper = () => loadUser()\n"
        "const config = { retries: 3 }\n",
        encoding="utf-8",
    )

    symbols = semantic_lookup(project, "find_symbol", query="User")
    names = {s["name"] for s in symbols["symbols"]}
    assert {"User", "loadUser"} <= names
    assert symbols["index"]["languages"]["typescript"] == 1

    all_symbols = semantic_lookup(project, "find_symbol")
    kinds = {s["name"]: s["kind"] for s in all_symbols["symbols"]}
    assert kinds["helper"] == "function"
    assert kinds["config"] == "variable"


def test_reference_context_privacy_default_and_cache_safety(tmp_path):
    project = tmp_path / "private"
    project.mkdir()
    secret_line = "SECRET_SNIPPET = 'do-not-cache-me'"
    (project / "secret.py").write_text(
        f"{secret_line}\n"
        "def use_secret():\n"
        "    return SECRET_SNIPPET\n",
        encoding="utf-8",
    )

    default_refs = semantic_lookup(project, "references", query="SECRET_SNIPPET")
    assert default_refs["total"] >= 2
    assert all(r.get("context", "") == "" for r in default_refs["references"])

    cache_path = cache_path_for(project)
    cache_text = cache_path.read_text(encoding="utf-8")
    cache_data = json.loads(cache_text)
    assert cache_data["version"].endswith(".v2")
    assert secret_line not in cache_text
    assert all(r.get("context", "") == "" for r in cache_data.get("references", []))

    opt_in_refs = semantic_lookup(project, "references", query="SECRET_SNIPPET", include_context=True)
    assert any(secret_line in r.get("context", "") for r in opt_in_refs["references"])
    assert secret_line not in cache_path.read_text(encoding="utf-8")
