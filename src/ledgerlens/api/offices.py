"""What each office's page says.

Three pages, one template, one place to edit the words. The alternative was three HTML files that
drift apart the first time the governance paragraph changes, which is the part they all share.

Every figure quoted here is one the suite or the evals establish: the forecast count and the bridge
residual from the FP&A evals, the planted faults from the interviews pack's seeded set, the invoice
lines from the rate-card drift run, and the total-cost example from the normalization tests. Nothing
on these pages is a number the warehouse has not produced.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Problem:
    """A sentence the visitor has said themselves, and the one line back."""

    quote: str
    answer: str


@dataclass(frozen=True)
class Proof:
    figure: str
    count: str  # the plain number the figure counts up from
    body: str


@dataclass(frozen=True)
class Office:
    slug: str
    name: str
    audience: str
    title: str
    description: str
    display: tuple[str, ...]  # the headline, one slab per line, the last one accented
    lede: str
    placeholder: str
    suggestions: tuple[str, ...]
    problems: tuple[Problem, ...]
    steps: tuple[tuple[str, str], ...]
    proofs: tuple[Proof, ...]
    workspace: str
    workspace_label: str


FPA = Office(
    slug="fpa",
    name="FP&A",
    audience="the CFO's office",
    title="FP&A that shows its working",
    description=(
        "Variance split into volume, mix, rate and timing, summing to the cent, every part traced to "
        "the payment events underneath it. Synthetic data."
    ),
    display=("Close is done.", "Cash still", "missed plan."),
    lede=(
        "A bridge that adds up, a forecast tested once, and a reason for every figure. Click any "
        "number and the query, the source tables and the row ids come back with it."
    ),
    placeholder="Ask about your month end...",
    suggestions=(
        "Why is net cash under plan?",
        "How good is the forecast?",
        "Where does this number come from?",
    ),
    problems=(
        Problem(
            quote="Close is done and I still cannot say why cash missed plan.",
            answer=(
                "Volume, mix, rate and timing, summing to the variance to the cent. The residual is "
                "rounding, and each part opens the rows it came from."
            ),
        ),
        Problem(
            quote="The forecast looked good in the deck and wrong in the quarter.",
            answer=(
                "One model per driver, chosen by cross-validation, then tested a single time on six "
                "frozen months. Where nothing beat seasonal naive, seasonal naive stays and the page says so."
            ),
        ),
        Problem(
            quote="Every month I argue about which variances matter.",
            answer=(
                "Thresholds are set once and applied by code: 5% or $50,000 at the statement level, "
                "3% or $25,000 at the operating level, with one-time items and reclasses screened first."
            ),
        ),
    ),
    steps=(
        ("Rebuild cash from the events", "Collections, claims and payouts, on the basis each belongs to."),
        ("Bridge actual to plan", "Four parts and a residual, reconciled against the ledger."),
        ("Forecast, then leave it alone", "The test set is touched once and the result reported as it came."),
        ("Hand over the rows", "Any figure resolves to the query and the row ids behind it."),
    ),
    proofs=(
        Proof(
            figure="8 of 12",
            count="8",
            body="Forecast series where the reviewed model beats seasonal naive on six frozen months.",
        ),
        Proof(
            figure="$0.00",
            count="0",
            body="Residual on the August cash bridge: the four parts explain all of it.",
        ),
    ),
    workspace="/app#/fpa",
    workspace_label="Open FP&A",
)

HIRING = Office(
    slug="hiring",
    name="HR",
    audience="the CHRO's office",
    title="Interview quality you can defend",
    description=(
        "Interviewers, panels and stages measured, never candidates, with the audit tables a regulator "
        "asks for. Synthetic data."
    ),
    display=("Legal wants", "a bias audit.", "You have ratings."),
    lede=(
        "The process is measured, not the person. No candidate score or ranking exists anywhere, and "
        "any group under five people is suppressed before it reaches a page."
    ),
    placeholder="Ask about your hiring loop...",
    suggestions=(
        "How is interviewer drift found?",
        "Why are small groups hidden?",
        "Does the score predict performance?",
    ),
    problems=(
        Problem(
            quote="Legal wants a bias audit and I have a spreadsheet of ratings.",
            answer=(
                "Selection rates by stage with the four-fifths screen, panel agreement, and the "
                "disclosure tables, all computed rather than asserted."
            ),
        ),
        Problem(
            quote="One interviewer is obviously lenient and I cannot prove it.",
            answer=(
                "Each interviewer against their own peers on the same roles, with the false-discovery "
                "correction that hundreds of comparisons require."
            ),
        ),
        Problem(
            quote="Does any of this actually predict who does well?",
            answer=(
                "Interview score against later performance, per role, with the interval shown. Where "
                "it spans zero the page says the interview does not predict, rather than rounding up."
            ),
        ),
    ),
    steps=(
        ("Measure the process", "Interviewers, panels and stages. Never a candidate, at any point."),
        ("Correct for the testing", "Hundreds of comparisons get one correction, so a flag means something."),
        (
            "Suppress the small groups",
            "Under five people, nothing is published, however the question is asked.",
        ),
        ("Leave an audit trail", "Every figure and every run stays queryable long after the decision."),
    ),
    proofs=(
        Proof(
            figure="4 of 4",
            count="4",
            body=(
                "Planted faults found: the lenient interviewer, the random panel, the adverse-impact "
                "stage and the non-predictive role."
            ),
        ),
        Proof(
            figure="5",
            count="5",
            body="People in a group, below which nothing is reported at all.",
        ),
    ),
    workspace="/app#/interviews",
    workspace_label="Open HR",
)

PROCUREMENT = Office(
    slug="procurement",
    name="Procurement",
    audience="the CPO's office",
    title="What a role actually costs",
    description=(
        "Billed titles mapped to standard roles, vendors compared on total cost, savings traced to "
        "invoice lines. Synthetic data."
    ),
    display=("Forty vendors.", "Six hundred titles.", "One real rate."),
    lede=(
        "Deals priced four different ways, normalised to one hourly basis, then the cheapest panel "
        "that still covers every role. Each step of the saving traces to the lines it came from."
    ),
    placeholder="Ask about your vendors...",
    suggestions=(
        "Where is the overbilling?",
        "How is the panel chosen?",
        "What would switching save?",
    ),
    problems=(
        Problem(
            quote="Forty vendors, six hundred job titles, and no idea what a role really costs.",
            answer=(
                "Titles mapped to standard roles, with every uncertain mapping sent for review rather "
                "than guessed."
            ),
        ),
        Problem(
            quote="One vendor quotes hourly, one monthly, one on outcomes.",
            answer=(
                "All of them normalised to USD per baseline hour, so $110 with a 20% productivity "
                "commitment is $88 and beats a $100 hourly rate."
            ),
        ),
        Problem(
            quote="I suspect we are billed above the card and cannot show it.",
            answer=(
                "Every invoice line checked against the contracted rate, with the overbilled and "
                "duplicated ones listed by line id."
            ),
        ),
    ),
    steps=(
        ("Map the titles", "Billed titles to standard roles, with the doubtful ones escalated."),
        ("Normalise the deals", "Hourly, monthly, fixed and outcome pricing onto one basis."),
        ("Check every line", "Invoices against the card they were supposed to follow."),
        ("Propose, never sign", "A panel change is a proposal until a named person approves it."),
    ),
    proofs=(
        Proof(
            figure="2,055",
            count="2055",
            body="Overbilled and duplicate invoice lines found and traced to their line ids.",
        ),
        Proof(
            figure="$88",
            count="88",
            body="Per baseline hour for a $110 outcome rate with a 20% commitment, under a $100 hourly deal.",
        ),
    ),
    workspace="/app#/ratecards",
    workspace_label="Open procurement",
)

OFFICES: dict[str, Office] = {office.slug: office for office in (FPA, HIRING, PROCUREMENT)}
