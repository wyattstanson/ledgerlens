"""Composition root: the one place that knows every pack. Core never imports a pack (tested)."""

from ledgerlens.core.registry import ToolRegistry
from ledgerlens.core.tools import CORE_TOOLS
from ledgerlens.fpa.tools import FPA_TOOLS
from ledgerlens.interviews.tools import INTERVIEW_TOOLS
from ledgerlens.ratecards.tools import RATECARD_TOOLS


def default_tool_registry() -> ToolRegistry:
    """Every tool shipped with LedgerLens."""
    return ToolRegistry((*CORE_TOOLS, *RATECARD_TOOLS, *INTERVIEW_TOOLS, *FPA_TOOLS))
