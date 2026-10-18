"""LedgerLens analyst app. Pack pages (app/pages/) arrive in Phase 5."""

import streamlit as st

from ledgerlens import __version__

st.set_page_config(page_title="LedgerLens", layout="wide")

st.title("LedgerLens")
st.caption(f"Governed analytics agents · v{__version__} · synthetic data only")

st.markdown(
    """
Every number on these pages is produced by deterministic code and carries lineage
to the query and rows behind it. The language model only plans and explains.

| Pack | Status |
|---|---|
| Rate Card Consolidation | Phase 2 |
| Interview Quality and Fairness Observatory | Phase 3 |
| Payments-aware FP&A | Phase 4 |
"""
)
