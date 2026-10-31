"""Every browser module must parse.

A syntax error in a page module is invisible to Python: the API serves the file with a 200, the
suite stays green, and the page renders an empty skeleton in the browser. `?? x || y` without
parentheses shipped exactly that way once. Node is the only parser we have for these files, so the
check runs where Node exists and says so where it does not.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from ledgerlens.api.main import STATIC_DIR

MODULES = sorted(STATIC_DIR.rglob("*.js"))


def test_there_are_modules_to_check():
    assert MODULES, "static modules moved; this guard is checking nothing"


@pytest.mark.parametrize("module", MODULES, ids=lambda p: p.name)
def test_a_browser_module_parses(module: Path, tmp_path: Path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed, so the modules cannot be parsed here")
    # --check reads the extension, not the content, so an .mjs copy gets the module grammar.
    copy = tmp_path / f"{module.stem}.mjs"
    copy.write_bytes(module.read_bytes())
    result = subprocess.run([node, "--check", str(copy)], check=False, capture_output=True, text=True)
    assert result.returncode == 0, f"{module.name} does not parse:\n{result.stderr}"
