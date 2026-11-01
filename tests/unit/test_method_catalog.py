"""The formula catalogue: every panel must actually reach the page.

The Method page renders one section per group in GROUPS and fills it with the models that name that
group. A model in a group with no section is built and then rendered nowhere, which no parse check
and no Python test would otherwise notice, because the page still loads and still looks complete.
"""

import re

import pytest

from ledgerlens.api.main import STATIC_DIR

METHOD = STATIC_DIR / "pages" / "method.js"
SOURCE = METHOD.read_text(encoding="utf-8")

# Each model is written as a block starting with its id, so one regex over the three fields pairs them.
MODELS = re.findall(r'\n    id: "([\w-]+)",\n    icon: "([\w-]+)",\n    group: "([^"]+)",', SOURCE)
GROUPS = re.findall(r'\n  \["([^"]+)", "([\w-]+)"\],', SOURCE)


def test_the_catalogue_was_found():
    assert len(MODELS) >= 20, f"only {len(MODELS)} models parsed; the file's shape changed"
    assert len(GROUPS) >= 2, "the group list did not parse"


def test_no_formula_is_declared_twice():
    ids = [model[0] for model in MODELS]
    assert len(ids) == len(set(ids)), f"duplicate ids: {sorted({i for i in ids if ids.count(i) > 1})}"


def test_every_formula_belongs_to_a_group_the_page_renders():
    rendered = {name for name, _ in GROUPS}
    orphans = {group for _, _, group in MODELS} - rendered
    assert not orphans, f"these groups have models but no section: {sorted(orphans)}"


def test_every_group_the_page_renders_has_formulas_in_it():
    used = {group for _, _, group in MODELS}
    empty = {name for name, _ in GROUPS} - used
    assert not empty, f"these sections would render empty: {sorted(empty)}"


@pytest.mark.parametrize("icon", sorted({icon for _, icon, _ in MODELS} | {icon for _, icon in GROUPS}))
def test_every_icon_is_a_phosphor_name(icon: str):
    # The page writes `ph-light ph-foo`; a name without the prefix silently renders nothing at all.
    assert icon.startswith("ph-"), icon


def test_the_workspace_loads_the_icon_font_it_uses():
    workspace = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert "phosphor-icons" in workspace, "the rail and the panels use ph-light classes"
    assert "ph-light" in workspace
