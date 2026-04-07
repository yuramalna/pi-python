"""Generate API reference pages for mkdocstrings.

This script is run by mkdocs-gen-files during the docs build.
It walks the pi_llm and pi_llm_agent source trees and generates one
reference page per public module.
"""

from pathlib import Path

import mkdocs_gen_files

PACKAGES = [
    {
        "name": "pi_llm",
        "src": Path("packages/pi_ai/src/pi_llm"),
        "doc_prefix": "pi_ai/reference",
    },
    {
        "name": "pi_llm_agent",
        "src": Path("packages/pi_agent/src/pi_llm_agent"),
        "doc_prefix": "pi_agent/reference",
    },
]

# Modules to skip (internal implementation details)
SKIP_MODULES = {
    "pi_llm.providers.openai_responses_shared",
    "pi_llm.providers.transform_messages",
    "pi_llm.utils.hash",
    "pi_llm.utils.json_parse",
    "pi_llm.utils.sanitize_unicode",
}

# Collect entries for index pages
index_entries: dict[str, list[tuple[str, str]]] = {}

for pkg in PACKAGES:
    src_root = pkg["src"]
    doc_prefix = pkg["doc_prefix"]
    entries: list[tuple[str, str]] = []

    for path in sorted(src_root.rglob("*.py")):
        # Build module path
        module_path = path.relative_to(src_root.parent)
        parts = list(module_path.with_suffix("").parts)

        # Skip __pycache__ and private modules
        if any(part.startswith("_") and part != "__init__" for part in parts):
            continue

        # Build the full module identifier
        if parts[-1] == "__init__":
            parts = parts[:-1]
            full_module = ".".join(parts)
        else:
            full_module = ".".join(parts)

        # Skip internal modules
        if full_module in SKIP_MODULES:
            continue

        if not full_module:
            continue

        # Use flat filenames for simplicity
        safe_name = full_module.replace(".", "_")
        doc_path = Path(doc_prefix) / f"{safe_name}.md"

        with mkdocs_gen_files.open(doc_path, "w") as fd:
            fd.write(f"# `{full_module}`\n\n")
            fd.write(f"::: {full_module}\n")

        mkdocs_gen_files.set_edit_path(doc_path, path)
        entries.append((full_module, f"{safe_name}.md"))

    index_entries[pkg["name"]] = entries

# Generate index pages for each package's reference section
for pkg in PACKAGES:
    doc_prefix = pkg["doc_prefix"]
    entries = index_entries[pkg["name"]]

    with mkdocs_gen_files.open(Path(doc_prefix) / "index.md", "w") as fd:
        fd.write(f"# {pkg['name']} API Reference\n\n")
        for module_name, filename in entries:
            fd.write(f"- [`{module_name}`]({filename})\n")
