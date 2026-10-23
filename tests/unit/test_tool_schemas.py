"""Every shipped tool schema stays inside what strict tool use accepts."""

from typing import Any

import pytest

from ledgerlens.packs import default_tool_registry

# JSON Schema keywords that strict tool schemas reject: a request carrying them fails before the model runs.
UNSUPPORTED = {
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
}


def keywords(node: Any) -> set[str]:
    if isinstance(node, dict):
        found = set(node) & UNSUPPORTED
        for key, value in node.items():
            if key != "properties":
                found |= keywords(value)
        for value in node.get("properties", {}).values():
            found |= keywords(value)
        return found
    if isinstance(node, list):
        return set().union(*(keywords(v) for v in node)) if node else set()
    return set()


@pytest.mark.parametrize("tool", list(default_tool_registry()), ids=lambda tool: tool.name)
def test_tool_schema_uses_only_strict_compatible_keywords(tool):
    assert keywords(tool.input_schema) == set()
