"""Compliance pack (BUILD_PROMPT 11.4): Markdown documents plus JSON, rendered from warehouse aggregates only.

These are documents, not legal advice. Each one carries a banner and the sources it relies on (verified
2026-10-19). Small cells are already suppressed in the aggregates; groups under the plan's minimum share
are listed as excluded. Thresholds in the text come from the run's own plan.
"""

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ledgerlens.core.db import connect
from ledgerlens.core.lineage import sha256_text
from ledgerlens.settings import REPO_ROOT

COMPLIANCE_DIR = REPO_ROOT / "data" / "compliance"
BANNER = (
    "> **Not legal advice.** Generated from synthetic data by LedgerLens to show the shape of the document. "
    "Have counsel review before any use."
)
RETRIEVED = "2026-10-19"
SAFE_NAME = re.compile(r"[A-Za-z0-9_-]+")
# The JSON export is an allow-list: a new column in the warehouse never reaches it by accident.
FAIRNESS_COLUMNS = (
    "role_id",
    "stage",
    "dimension",
    '"group"',
    "n",
    "selected",
    "rate",
    "impact_ratio",
    "reference_group",
    "excluded_small_share",
    "tested",
    "h",
    "h_ci_low",
    "h_ci_high",
    "q",
    "flagged",
    "suppressed",
)
INTERSECTIONAL_COLUMNS = (
    "role_id",
    "stage",
    "sex",
    "race_ethnicity",
    "n",
    "selected",
    "rate",
    "impact_ratio",
    "excluded_small_share",
    "suppressed",
)
META_COLUMNS = (
    "analysis_run_id",
    "plan_version",
    "plan_sha256",
    "tests_in_family",
    "min_group_n",
    "min_group_share",
    "impact_ratio_threshold",
    "alpha",
)
SOURCES = {
    "nyc_ll144": [
        "https://www.nyc.gov/assets/dca/downloads/pdf/about/DCWP-AEDT-FAQ.pdf",
        "https://rules.cityofnewyork.us/wp-content/uploads/2023/04/DCWP-NOA-for-Use-of-Automated-Employment-Decisionmaking-Tools-2.pdf",
    ],
    "colorado_sb26_189": ["https://leg.colorado.gov/bills/sb26-189", "https://coag.gov/ai/"],
    "illinois_hb3773": ["https://www.ilga.gov/legislation/publicacts/fulltext.asp?Name=103-0804"],
    "eu_ai_act": ["https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai"],
}


@dataclass(frozen=True)
class Pack:
    run_id: str
    files: dict[str, str] = field(default_factory=dict)

    @property
    def digest(self) -> str:
        return sha256_text(json.dumps(self.files, sort_keys=True))


def _sources(key: str) -> str:
    return "\n".join(["", "## Sources", "", *[f"- {url} (retrieved {RETRIEVED})" for url in SOURCES[key]]])


def _rows(conn: Any, sql: str) -> list[dict[str, Any]]:
    cursor = conn.execute(sql)
    columns = [d[0] for d in cursor.description]
    return [dict(zip(columns, r, strict=True)) for r in cursor.fetchall()]


def _fmt(value: Any, pct: bool = False) -> str:
    if value is None:
        return "suppressed"
    return f"{value:.1%}" if pct else f"{value:.3f}"


def _ratio(row: dict[str, Any], share: float) -> str:
    if row["excluded_small_share"]:
        return f"excluded (<{share:.0%})"
    if row["suppressed"]:
        return "suppressed"
    return "not compared" if row["impact_ratio"] is None else f"{row['impact_ratio']:.3f}"


def _ll144(fair: list[dict], cross: list[dict], meta: dict) -> str:
    share, floor = meta["min_group_share"], meta["min_group_n"]
    lines = [
        "# NYC Local Law 144-style bias audit table",
        "",
        BANNER,
        "",
        "Selection rate and impact ratio per stage, by sex, by race/ethnicity, and intersectionally. "
        "Impact ratio = a category's selection rate / the highest category's rate. "
        f"Categories under {share:.0%} of the data are excluded from impact ratios, as the DCWP rules allow. "
        f"Cells under {floor} people are suppressed, with the cells that would reveal them by subtraction. "
        "Unknown is reported but never compared.",
        "",
    ]
    for role, stage in sorted({(r["role_id"], r["stage"]) for r in fair}):
        lines += [f"## {role} / {stage}", ""]
        for dim, label in (("sex", "Sex"), ("race_ethnicity", "Race/ethnicity")):
            lines += [
                f"**{label}**",
                "",
                "| Category | Applicants | Selected | Selection rate | Impact ratio |",
                "|---|---|---|---|---|",
            ]
            excluded = []
            for r in (
                x for x in fair if x["role_id"] == role and x["stage"] == stage and x["dimension"] == dim
            ):
                if r["excluded_small_share"]:
                    excluded.append(r["group"])
                n = "suppressed" if r["n"] is None else str(r["n"])
                sel = "suppressed" if r["selected"] is None else str(r["selected"])
                lines.append(
                    f"| {r['group']} | {n} | {sel} | {_fmt(r['rate'], pct=True)} | {_ratio(r, share)} |"
                )
            lines += ["", f"Excluded categories: {', '.join(excluded) or 'none'}", ""]
        lines += [
            "**Intersectional (sex x race/ethnicity)**",
            "",
            "| Sex | Race/ethnicity | Applicants | Selection rate | Impact ratio |",
            "|---|---|---|---|---|",
        ]
        for r in (x for x in cross if x["role_id"] == role and x["stage"] == stage):
            n = "suppressed" if r["n"] is None else str(r["n"])
            ratio = _ratio(r, share)
            lines.append(
                f"| {r['sex']} | {r['race_ethnicity']} | {n} | {_fmt(r['rate'], pct=True)} | {ratio} |"
            )
        lines.append("")
    return "\n".join(lines) + _sources("nyc_ll144")


def _colorado() -> str:
    return "\n".join(
        [
            "# Colorado SB 26-189 (automated decision-making technology): notice templates",
            "",
            BANNER,
            "",
            "Applies from 1 January 2027.",
            "",
            "## 1. Pre-use notice (clear and conspicuous, at points of applicant interaction)",
            "",
            "We use automated decision-making technology to help evaluate applications for this role. "
            "It assesses [describe inputs] to inform [describe decision]. A trained person reviews its "
            "output before any decision.",
            "",
            "## 2. Adverse decision explanation (within 30 days of the decision)",
            "",
            "Your application for [role] was not advanced on [date]. Automated decision-making "
            "technology was used to [describe its role]. The main factors were: [plain-language factors].",
            "",
            "## 3. Your rights",
            "",
            "- Access the personal data used in this decision, and correct anything factually incorrect.",
            "- Request meaningful human review and reconsideration. Reviewer: [name/role], trained and "
            "authorized to override the decision. Contact: [channel].",
        ]
    ) + _sources("colorado_sb26_189")


def _illinois(meta: dict, fair: list[dict]) -> str:
    flagged = [r for r in fair if r.get("flagged")]
    lines = [
        "# Illinois HB 3773 (Public Act 103-0804): outcome monitoring summary",
        "",
        BANNER,
        "",
        "In force since 1 January 2026. Employers must give notice when AI is used in covered employment "
        "decisions, must not use AI with a discriminatory effect on protected classes, and must not use zip "
        "codes as a proxy for them. Implementing rules from the Department of Human Rights are pending.",
        "",
        f"- Tests run this period: {meta['tests_in_family']} (calibration and fairness, one BH family)",
        f"- Stage/group cells with adverse impact (ratio < {meta['impact_ratio_threshold']:.2f} and "
        f"BH q < {meta['alpha']}): {len(flagged)}",
    ]
    lines += [
        f"  - {r['role_id']} / {r['stage']} / {r['dimension']} = {r['group']}: "
        f"impact ratio {_fmt(r['impact_ratio'])}"
        for r in flagged
    ]
    lines += [
        "- Zip code is not an input to any LedgerLens interview metric.",
        "- Notice to applicants and employees: [link to notice].",
    ]
    return "\n".join(lines) + _sources("illinois_hb3773")


def _eu() -> str:
    items = [
        "Risk management system for the recruitment use case",
        "Data governance: training and test data quality",
        "Technical documentation kept up to date",
        "Automatic logging of events (LedgerLens audit tables)",
        "Transparency information for deployers and candidates",
        "Human oversight: approvals by named humans",
        "Accuracy, robustness and cybersecurity measures",
        "Post-market monitoring plan",
    ]
    return "\n".join(
        [
            "# EU AI Act readiness checklist (recruitment = Annex III high-risk)",
            "",
            BANNER,
            "",
            "High-risk obligations for Annex III systems apply from 2 December 2027.",
            "",
            *[f"- [ ] {item}" for item in items],
        ]
    ) + _sources("eu_ai_act")


def render_pack(db_path: Path) -> Pack:
    with connect(db_path) as conn:
        meta = _rows(conn, f"select {', '.join(META_COLUMNS)} from interviews.run_meta")[0]  # noqa: S608
        fair = _rows(
            conn,
            f"select {', '.join(FAIRNESS_COLUMNS)} from interviews.fairness "  # noqa: S608
            'order by role_id, stage, dimension, "group"',
        )
        cross = _rows(
            conn,
            f"select {', '.join(INTERSECTIONAL_COLUMNS)} from interviews.intersectional "  # noqa: S608
            "order by role_id, stage, sex, race_ethnicity",
        )
    data = {
        "run_id": meta["analysis_run_id"],
        "plan_version": meta["plan_version"],
        "plan_sha256": meta["plan_sha256"],
        "fairness": fair,
        "intersectional": cross,
    }
    files = {
        "ll144_bias_audit.md": _ll144(fair, cross, meta),
        "colorado_sb26_189_notices.md": _colorado(),
        "illinois_hb3773_monitoring.md": _illinois(meta, fair),
        "eu_ai_act_checklist.md": _eu(),
        "compliance_data.json": json.dumps(data, indent=2, default=str),
    }
    return Pack(run_id=str(meta["analysis_run_id"]), files=files)


def write_pack(pack: Pack, root: Path, name: str) -> Path:
    """Write the pack to ``root/name`` whole or not at all: into a staging folder, then one rename."""
    if not SAFE_NAME.fullmatch(name):
        raise ValueError(f"unsafe export folder name {name!r}")
    folder, staging = root / name, root / f".{name}.staging"
    if folder.exists():
        raise FileExistsError(f"{folder} already exists")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    for file_name, content in pack.files.items():
        (staging / file_name).write_text(content, encoding="utf-8")
    staging.rename(folder)
    return folder
