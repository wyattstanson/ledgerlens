/* The method page: the arithmetic behind every figure, with the inputs live.

   A consultancy shows you the conclusion. This shows the equation, lets you move its inputs, and
   recomputes in front of you. The worked examples run in the browser on the numbers you set; the
   figures on the pack pages come from the governed tools and carry lineage, which the panels say. */

import { card, el, fmt } from "../lib.js";

export const title = "Method";

const GROUPS = [
  ["FP&A", "ph-chart-line-up"],
  ["HR", "ph-users"],
  ["Procurement", "ph-receipt"],
  ["Governance", "ph-shield-check"],
];

/* ---------- Formula definitions ----------
   Each has the equation as it appears in the spec, the inputs you can move, and the arithmetic. */

const MODELS = [
  {
    id: "net-cash",
    icon: "ph-wallet",
    group: "FP&A",
    name: "Net operating cash",
    equation: "net_operating_cash = collected_premium - claims_paid - lae - opex",
    why: "The driver tree rebuilds cash from the payment events underneath it, so a miss can be attributed rather than just reported.",
    inputs: [
      { key: "collected", label: "Collected premium", value: 3797090, step: 10000, unit: "USD" },
      { key: "claims", label: "Claims paid", value: 1795905, step: 10000, unit: "USD" },
      { key: "lae", label: "Loss adjustment expense", value: 4270, step: 500, unit: "USD" },
      { key: "opex", label: "Operating expense", value: 873704, step: 10000, unit: "USD" },
    ],
    compute: (v) => [
      { label: "Net operating cash", value: v.collected - v.claims - v.lae - v.opex, unit: "USD" },
    ],
  },
  {
    id: "collections",
    icon: "ph-hand-coins",
    group: "FP&A",
    name: "Collected premium",
    equation: "collected = Σ over methods ( due × method_mix × on_time_rate ) + late_collections",
    why: "Collections depend on how customers pay, not just on what was billed. Change the mix and the cash moves.",
    inputs: [
      { key: "due", label: "Due premium", value: 4000000, step: 100000, unit: "USD" },
      { key: "cardMix", label: "Card share", value: 0.34, step: 0.01, unit: "share", max: 1 },
      { key: "cardRate", label: "Card paid on time", value: 0.93, step: 0.01, unit: "share", max: 1 },
      { key: "achMix", label: "ACH share", value: 0.3, step: 0.01, unit: "share", max: 1 },
      { key: "achRate", label: "ACH paid on time", value: 0.97, step: 0.01, unit: "share", max: 1 },
      { key: "late", label: "Late cash from prior months", value: 180000, step: 10000, unit: "USD" },
    ],
    compute: (v) => {
      const card = v.due * v.cardMix * v.cardRate;
      const ach = v.due * v.achMix * v.achRate;
      return [
        { label: "From card", value: card, unit: "USD" },
        { label: "From ACH", value: ach, unit: "USD" },
        { label: "Collected (these two methods plus late cash)", value: card + ach + v.late, unit: "USD" },
      ];
    },
  },
  {
    id: "bridge",
    icon: "ph-chart-bar-horizontal",
    group: "FP&A",
    name: "Variance bridge",
    equation: "volume = (ΣQa - ΣQp) × P̄p    mix = Σ (share_a - share_p) × ΣQa × Pp    rate = Σ (Pa - Pp) × Qa",
    why: "Actual minus plan split into parts that sum to the whole. A property test proves the parts add to the variance to the cent.",
    inputs: [
      { key: "qa", label: "Actual quantity", value: 1200, step: 25, unit: "count" },
      { key: "qp", label: "Plan quantity", value: 1000, step: 25, unit: "count" },
      { key: "pa", label: "Actual rate", value: 92, step: 1, unit: "USD" },
      { key: "pp", label: "Plan rate", value: 100, step: 1, unit: "USD" },
    ],
    compute: (v) => {
      const volume = (v.qa - v.qp) * v.pp;
      const rate = (v.pa - v.pp) * v.qa;
      const total = v.qa * v.pa - v.qp * v.pp;
      return [
        { label: "Volume", value: volume, unit: "USD" },
        { label: "Rate", value: rate, unit: "USD" },
        { label: "Residual (one segment, so mix is zero)", value: total - volume - rate, unit: "USD" },
        { label: "Total variance", value: total, unit: "USD", strong: true },
      ];
    },
  },
  {
    id: "mase",
    icon: "ph-target",
    group: "FP&A",
    name: "Forecast error, MASE",
    equation: "MASE = mean|forecast - actual| ÷ mean|seasonal_naive - actual|",
    why: "Scaled against the naive forecast, so 1.00 means no better than repeating last season. Below 1 earns its place.",
    inputs: [
      { key: "err", label: "Mean absolute error of the model", value: 64000, step: 1000, unit: "USD" },
      { key: "naive", label: "Mean absolute error of seasonal naive", value: 100000, step: 1000, unit: "USD" },
    ],
    compute: (v) => {
      const mase = v.naive === 0 ? null : v.err / v.naive;
      return [
        { label: "MASE", value: mase, unit: "ratio", decimals: 2, strong: true },
        { label: mase !== null && mase < 1 ? "Beats the baseline" : "Does not beat the baseline", value: null },
      ];
    },
  },
  {
    id: "impact",
    icon: "ph-scales",
    group: "HR",
    name: "Impact ratio, the four-fifths rule",
    equation: "impact_ratio = selection_rate_group ÷ selection_rate_highest_group",
    why: "Below 0.80 is the threshold US guidance uses. Measured per stage, on the process, never on a person.",
    inputs: [
      { key: "passA", label: "Group A passed", value: 76, step: 1, unit: "count" },
      { key: "totalA", label: "Group A assessed", value: 171, step: 1, unit: "count" },
      { key: "passB", label: "Group B passed", value: 145, step: 1, unit: "count" },
      { key: "totalB", label: "Group B assessed", value: 202, step: 1, unit: "count" },
    ],
    compute: (v) => {
      const a = v.totalA ? v.passA / v.totalA : 0;
      const b = v.totalB ? v.passB / v.totalB : 0;
      const highest = Math.max(a, b);
      const ratio = highest ? Math.min(a, b) / highest : 0;
      const small = v.totalA < 5 || v.totalB < 5;
      return [
        { label: "Group A selection rate", value: a, unit: "rate" },
        { label: "Group B selection rate", value: b, unit: "rate" },
        { label: "Impact ratio", value: ratio, unit: "ratio", decimals: 2, strong: true },
        {
          label: small
            ? "Suppressed: a group under five people is never published"
            : ratio < 0.8
              ? "Below four fifths, so this cell is flagged"
              : "At or above four fifths",
          value: null,
        },
      ];
    },
  },
  {
    id: "bh",
    icon: "ph-funnel",
    group: "HR",
    name: "Benjamini-Hochberg correction",
    equation: "keep test i (ranked by p) while p(i) ≤ (i ÷ m) × α",
    why: "Run 469 tests at 5% and about 23 look significant by chance. One correction across every test in the run, registered before it runs.",
    inputs: [
      { key: "m", label: "Tests in the family", value: 469, step: 1, unit: "count" },
      { key: "alpha", label: "Alpha", value: 0.05, step: 0.01, unit: "ratio", max: 1 },
      { key: "rank", label: "Rank of this test", value: 5, step: 1, unit: "count" },
    ],
    compute: (v) => {
      const threshold = v.m ? (v.rank / v.m) * v.alpha : 0;
      return [
        { label: "False positives expected with no correction", value: v.m * v.alpha, unit: "count", decimals: 1 },
        { label: "This test's threshold", value: threshold, unit: "ratio", decimals: 6, strong: true },
      ];
    },
  },
  {
    id: "normalise",
    icon: "ph-equals",
    group: "Procurement",
    name: "Rate normalisation",
    equation: "FTE: monthly ÷ standard_hours      Outcome: rate × (1 - productivity_commitment)",
    why: "The cheapest hourly rate is often not the cheapest work. Normalising to one basis is what exposes that.",
    inputs: [
      { key: "monthly", label: "Monthly FTE rate", value: 14400, step: 200, unit: "USD" },
      { key: "hours", label: "Standard hours a month", value: 160, step: 5, unit: "count" },
      { key: "outcome", label: "Outcome-based rate", value: 110, step: 5, unit: "USD" },
      { key: "commitment", label: "Productivity commitment", value: 0.2, step: 0.05, unit: "share", max: 1 },
      { key: "tm", label: "Time and materials rate", value: 100, step: 5, unit: "USD" },
    ],
    compute: (v) => {
      const fte = v.hours ? v.monthly / v.hours : 0;
      const effective = v.outcome * (1 - v.commitment);
      const rates = [
        { name: "the FTE deal", rate: fte },
        { name: "the outcome deal", rate: effective },
        { name: "time and materials", rate: v.tm },
      ].sort((x, y) => x.rate - y.rate);
      return [
        { label: "FTE, per hour", value: fte, unit: "USD", decimals: 2 },
        { label: "Outcome deal, effective", value: effective, unit: "USD", decimals: 2 },
        { label: `Cheapest is ${rates[0].name}`, value: null, strong: true },
      ];
    },
  },
  {
    id: "panel",
    icon: "ph-graph",
    group: "Procurement",
    name: "Panel optimiser",
    equation: "minimise Σ p(v,r) × h(v,r)   subject to   Σ_v h(v,r) = D(r),   h(v,r) ≤ D(r) × keep(v),   Σ keep(v) ≤ N",
    why: "Every role's demand must be met, hours only go to vendors you keep, and the panel has a size limit. Solved exactly, not by ranking.",
    inputs: [
      { key: "demand", label: "Hours needed for this role", value: 10000, step: 500, unit: "count" },
      { key: "cheap", label: "Cheapest kept vendor rate", value: 33.42, step: 0.5, unit: "USD" },
      { key: "incumbent", label: "Incumbent rate", value: 41.1, step: 0.5, unit: "USD" },
      { key: "cap", label: "Most of the demand one vendor may hold", value: 0.6, step: 0.05, unit: "share", max: 1 },
    ],
    compute: (v) => {
      const atCheap = Math.min(1, v.cap) * v.demand;
      const rest = v.demand - atCheap;
      const cost = atCheap * v.cheap + rest * v.incumbent;
      return [
        { label: "Cost at the concentration cap", value: cost, unit: "USD" },
        { label: "Cost if the incumbent kept it all", value: v.demand * v.incumbent, unit: "USD" },
        { label: "Saving", value: v.demand * v.incumbent - cost, unit: "USD", strong: true },
      ];
    },
  },
  {
    id: "materiality",
    icon: "ph-warning-diamond",
    group: "FP&A",
    name: "Materiality thresholds",
    equation: "material  ⟺  |v| ≥ usd_floor  or  |v| ≥ pct_floor × |plan|        immaterial  ⟺  |v| < 10,000  and  |v| < 2% × |plan|",
    why: "ADR 0001 sets the floors once: 5% or $50,000 at the statement level, 3% or $25,000 at the operating level. Code applies them, so nobody argues a figure into being immaterial after seeing it.",
    inputs: [
      { key: "actual", label: "Actual", value: 1100000, step: 10000, unit: "USD" },
      { key: "plan", label: "Plan", value: 1000000, step: 10000, unit: "USD" },
      { key: "pct", label: "Percentage floor", value: 0.03, step: 0.005, unit: "share", max: 0.1 },
      { key: "usd", label: "Dollar floor", value: 25000, step: 1000, unit: "USD" },
    ],
    compute: (v) => {
      const variance = v.actual - v.plan;
      const size = Math.abs(variance);
      const material = size >= v.usd || (v.plan > 0 && size >= v.pct * Math.abs(v.plan));
      const immaterial = size < 10000 && (v.plan === 0 || size < 0.02 * Math.abs(v.plan));
      return [
        { label: "Variance", value: variance, unit: "USD" },
        { label: "Percentage of plan", value: v.plan ? (size / Math.abs(v.plan)) * 100 : 0, decimals: 1, unit: "pct" },
        { label: "Dollar floor reached", value: size >= v.usd ? 1 : 0, text: size >= v.usd ? "yes" : "no" },
        {
          label: "Status",
          text: material ? "material" : immaterial ? "immaterial" : "watch",
          strong: true,
        },
      ];
    },
  },
  {
    id: "reclass",
    icon: "ph-arrows-left-right",
    group: "FP&A",
    name: "The reclassification screen",
    equation: "reclass  ⟺  |v + v_other| ≤ 0.02 × |v|      for another expense measure in the same month",
    why: "A ledger reclass moves an amount between two expense accounts, so it shows up as two equal and opposite variances. Compute checks for the other half before flagging either, which is what a reviewer would do first.",
    inputs: [
      { key: "variance", label: "This variance", value: 120000, step: 5000, unit: "USD" },
      { key: "other", label: "The other expense measure's variance", value: -119000, step: 5000, unit: "USD", min: -400000 },
      { key: "tol", label: "Tolerance", value: 0.02, step: 0.005, unit: "share", max: 0.1 },
    ],
    compute: (v) => {
      const residual = Math.abs(v.variance + v.other);
      const allowed = v.tol * Math.abs(v.variance);
      return [
        { label: "Left over after the offset", value: residual, unit: "USD" },
        { label: "Tolerance in dollars", value: allowed, unit: "USD" },
        { label: "Verdict", text: residual <= allowed ? "reclass, not flagged" : "a real variance", strong: true },
      ];
    },
  },
  {
    id: "seasonal-naive",
    icon: "ph-waveform",
    group: "FP&A",
    name: "The seasonal naive scale",
    equation: "scale = mean | y(t) - y(t - m) |   over the training rows,   m = season length",
    why: "This is the denominator in MASE, and it is what the forecast has to beat. A model that cannot beat repeating last season is not kept, however good its chart looks.",
    inputs: [
      { key: "swing", label: "Typical month-on-month swing", value: 48000, step: 2000, unit: "USD" },
      { key: "error", label: "Your model's mean absolute error", value: 31000, step: 1000, unit: "USD" },
    ],
    compute: (v) => {
      const mase = v.swing ? v.error / v.swing : NaN;
      return [
        { label: "Scale (seasonal naive error)", value: v.swing, unit: "USD" },
        { label: "MASE", value: mase, decimals: 2 },
        { label: "Verdict", text: mase < 1 ? "beats seasonal naive" : "seasonal naive stays", strong: true },
      ];
    },
  },
  {
    id: "scenario",
    icon: "ph-swap",
    group: "FP&A",
    name: "Payout channel scenario",
    equation: "cash_week(w) = Σ payouts approved in week w - lag(channel)      lae = Σ amount × handling(channel)",
    why: "A scenario changes driver inputs, never outputs. Move a share of payouts from check to instant and the cash leaves sooner and the handling costs less, both as expected values, so the answer is the same every time.",
    inputs: [
      { key: "payouts", label: "Claim payouts in the window", value: 1800000, step: 50000, unit: "USD" },
      { key: "share", label: "Share moved to instant", value: 0.3, step: 0.05, unit: "share", max: 1 },
      { key: "checkLag", label: "Check lag, days", value: 9, step: 1, unit: "count", max: 30 },
      { key: "instantLag", label: "Instant lag, days", value: 1, step: 1, unit: "count", max: 30 },
      { key: "checkCost", label: "Check handling cost per dollar", value: 0.004, step: 0.0005, unit: "rate", max: 0.02 },
      { key: "instantCost", label: "Instant handling cost per dollar", value: 0.0015, step: 0.0005, unit: "rate", max: 0.02 },
    ],
    compute: (v) => {
      const moved = v.payouts * v.share;
      const daysEarlier = (v.checkLag - v.instantLag) * v.share;
      const lae = (v.payouts - moved) * v.checkCost + moved * v.instantCost;
      const was = v.payouts * v.checkCost;
      return [
        { label: "Payouts moved", value: moved, unit: "USD" },
        { label: "Average days earlier the cash leaves", value: daysEarlier, decimals: 1 },
        { label: "Loss adjustment expense now", value: lae, unit: "USD" },
        { label: "Change in handling cost", value: lae - was, unit: "USD", strong: true },
      ];
    },
  },
  {
    id: "selection-rate",
    icon: "ph-funnel-simple",
    group: "HR",
    name: "Selection rate by stage",
    equation: "rate(group, stage) = passed / entered        impact_ratio = rate(group) / max rate among eligible groups",
    why: "Measured per role, per stage, per dimension. Every group that applied gets a row even when nobody is left, so a missing line never tells a reader that a group vanished.",
    inputs: [
      { key: "enteredA", label: "Group A entered", value: 120, step: 1, unit: "count" },
      { key: "passedA", label: "Group A passed", value: 48, step: 1, unit: "count" },
      { key: "enteredB", label: "Group B entered", value: 90, step: 1, unit: "count" },
      { key: "passedB", label: "Group B passed", value: 27, step: 1, unit: "count" },
    ],
    compute: (v) => {
      const a = v.enteredA ? v.passedA / v.enteredA : 0;
      const b = v.enteredB ? v.passedB / v.enteredB : 0;
      const top = Math.max(a, b);
      return [
        { label: "Group A rate", value: a * 100, decimals: 1, unit: "pct" },
        { label: "Group B rate", value: b * 100, decimals: 1, unit: "pct" },
        { label: "Lower group's impact ratio", value: top ? Math.min(a, b) / top : 0, decimals: 3, strong: true },
      ];
    },
  },
  {
    id: "cohens-h",
    icon: "ph-arrows-out-line-horizontal",
    group: "HR",
    name: "Cohen's h, the effect size for two rates",
    equation: "h = 2 · arcsin √p₁ - 2 · arcsin √p₂",
    why: "A ratio alone cannot say whether a gap is large. The arcsine transform makes the difference between 2% and 4% comparable with the one between 50% and 52%, which a plain subtraction does not.",
    inputs: [
      { key: "p1", label: "Higher rate", value: 0.4, step: 0.01, unit: "share", max: 1 },
      { key: "p2", label: "Lower rate", value: 0.3, step: 0.01, unit: "share", max: 1 },
    ],
    compute: (v) => {
      const h = 2 * Math.asin(Math.sqrt(v.p1)) - 2 * Math.asin(Math.sqrt(v.p2));
      const size = Math.abs(h);
      return [
        { label: "Difference in rates", value: (v.p1 - v.p2) * 100, decimals: 1, unit: "pct" },
        { label: "Cohen's h", value: h, decimals: 3 },
        {
          label: "Conventional reading",
          text: size < 0.2 ? "negligible" : size < 0.5 ? "small" : size < 0.8 ? "medium" : "large",
          strong: true,
        },
      ];
    },
  },
  {
    id: "cohens-d",
    icon: "ph-ruler",
    group: "HR",
    name: "Cohen's d, the effect size for two means",
    equation: "d = (mean_a - mean_b) / s_pooled,      s_pooled = √( ((n_a-1)s_a² + (n_b-1)s_b²) / (n_a + n_b - 2) )",
    why: "How far an interviewer sits from their peers, in standard deviations rather than in score points, so roles scored on different scales can be compared.",
    inputs: [
      { key: "meanA", label: "This interviewer's mean score", value: 3.6, step: 0.05, unit: "ratio", max: 5 },
      { key: "meanB", label: "Peer mean score", value: 3.1, step: 0.05, unit: "ratio", max: 5 },
      { key: "sd", label: "Pooled standard deviation", value: 0.62, step: 0.02, unit: "ratio", max: 2 },
    ],
    compute: (v) => {
      const d = v.sd ? (v.meanA - v.meanB) / v.sd : NaN;
      const size = Math.abs(d);
      return [
        { label: "Gap in score points", value: v.meanA - v.meanB, decimals: 2 },
        { label: "Cohen's d", value: d, decimals: 2 },
        {
          label: "Conventional reading",
          text: size < 0.2 ? "negligible" : size < 0.5 ? "small" : size < 0.8 ? "medium" : "large",
          strong: true,
        },
      ];
    },
  },
  {
    id: "kappa",
    icon: "ph-users-three",
    group: "HR",
    name: "Panel agreement, Fleiss' kappa",
    equation: "κ = (P̄ - P̄ₑ) / (1 - P̄ₑ)      P̄ = observed agreement,   P̄ₑ = agreement expected by chance",
    why: "Four interviewers who all say hire agree completely, but if everyone says hire to everyone that agreement is worth nothing. Kappa subtracts the agreement chance alone would produce.",
    inputs: [
      { key: "observed", label: "Observed agreement", value: 0.78, step: 0.01, unit: "share", max: 1 },
      { key: "chance", label: "Agreement expected by chance", value: 0.55, step: 0.01, unit: "share", max: 1 },
    ],
    compute: (v) => {
      const kappa = v.chance < 1 ? (v.observed - v.chance) / (1 - v.chance) : NaN;
      return [
        { label: "Fleiss' kappa", value: kappa, decimals: 3 },
        {
          label: "Conventional reading",
          text:
            kappa < 0.2 ? "slight" : kappa < 0.4 ? "fair" : kappa < 0.6 ? "moderate" : kappa < 0.8 ? "substantial" : "almost perfect",
          strong: true,
        },
      ];
    },
  },
  {
    id: "drift-z",
    icon: "ph-standard-definition",
    group: "HR",
    name: "Interviewer drift, the peer z-score",
    equation: "z = (mean_interviewer - mean of peer means) / sd of peer means,      peers ≥ 2",
    why: "An interviewer is compared with the other people who interviewed for the same role on the same competency, never with a global average. Fewer than two peers gives no spread to measure against, so no score is produced at all.",
    inputs: [
      { key: "mine", label: "This interviewer's mean", value: 3.8, step: 0.05, unit: "ratio", max: 5 },
      { key: "peerMean", label: "Mean of the peer means", value: 3.2, step: 0.05, unit: "ratio", max: 5 },
      { key: "peerSd", label: "Spread of the peer means", value: 0.24, step: 0.01, unit: "ratio", max: 2 },
      { key: "peers", label: "Number of peers", value: 6, step: 1, unit: "count", max: 30 },
    ],
    compute: (v) => {
      if (v.peers < 2) return [{ label: "Result", text: "fewer than two peers, so no score is produced", strong: true }];
      const z = v.peerSd ? (v.mine - v.peerMean) / v.peerSd : NaN;
      return [
        { label: "Peer z-score", value: z, decimals: 2 },
        { label: "Direction", text: z > 0 ? "more lenient than peers" : "stricter than peers" },
        { label: "Beyond two standard deviations", text: Math.abs(z) >= 2 ? "yes, worth a look" : "no", strong: true },
      ];
    },
  },
  {
    id: "spearman",
    icon: "ph-chart-scatter",
    group: "HR",
    name: "Does the interview predict performance?",
    equation: "ρ = Pearson correlation of the ranks of x and y,      reported with a bootstrap interval",
    why: "Rank correlation between interview score and later performance, per role. Where the interval spans zero the page says the interview does not predict, instead of rounding a weak signal up into a claim.",
    inputs: [
      { key: "rho", label: "Observed rank correlation", value: 0.31, step: 0.01, unit: "ratio", max: 1, min: -1 },
      { key: "half", label: "Half-width of the 95% interval", value: 0.18, step: 0.01, unit: "ratio", max: 1 },
    ],
    compute: (v) => {
      const low = v.rho - v.half;
      const high = v.rho + v.half;
      return [
        { label: "Interval low", value: low, decimals: 2 },
        { label: "Interval high", value: high, decimals: 2 },
        {
          label: "What the page says",
          text: low > 0 ? "the interview predicts performance" : "the interval spans zero, so it does not predict",
          strong: true,
        },
      ];
    },
  },
  {
    id: "suppression",
    icon: "ph-eye-slash",
    group: "HR",
    name: "Small-group suppression",
    equation: "publish  ⟺  n ≥ 5   and   n / stage_total ≥ min_group_share",
    why: "Under five people nothing is published, however the question is asked. A group too small a share of the stage is left out of impact ratios and said to be excluded, so its absence is on the page rather than hidden in it.",
    inputs: [
      { key: "n", label: "People in the group", value: 7, step: 1, unit: "count", max: 60 },
      { key: "total", label: "People at this stage", value: 180, step: 5, unit: "count" },
      { key: "minShare", label: "Minimum share of the stage", value: 0.02, step: 0.005, unit: "share", max: 0.2 },
    ],
    compute: (v) => {
      const share = v.total ? v.n / v.total : 0;
      const bigEnough = v.n >= 5;
      const shareEnough = share >= v.minShare;
      return [
        { label: "Share of the stage", value: share * 100, decimals: 2, unit: "pct" },
        { label: "At least five people", text: bigEnough ? "yes" : "no" },
        {
          label: "Published",
          text: !bigEnough ? "no, suppressed" : shareEnough ? "yes" : "counted, but left out of impact ratios",
          strong: true,
        },
      ];
    },
  },
  {
    id: "overbill",
    icon: "ph-receipt-x",
    group: "Procurement",
    name: "Overbilling against the card",
    equation: "overbilled  ⟺  billed_rate / card_rate - 1 > threshold       excess = (billed_rate - card_rate) × quantity",
    why: "Compared in the card's own currency, so a currency move can never create a flag. The threshold is 2% and exactly 2% does not count, which is a floating-point trap the code handles explicitly.",
    inputs: [
      { key: "billed", label: "Billed rate", value: 104, step: 0.5, unit: "USD" },
      { key: "card", label: "Contracted card rate", value: 100, step: 0.5, unit: "USD" },
      { key: "hours", label: "Hours on the line", value: 160, step: 10, unit: "count" },
      { key: "threshold", label: "Threshold", value: 0.02, step: 0.005, unit: "share", max: 0.2 },
    ],
    compute: (v) => {
      const ratio = v.card ? v.billed / v.card - 1 : 0;
      const over = ratio > v.threshold + 1e-9;
      return [
        { label: "Above the card by", value: ratio * 100, decimals: 2, unit: "pct" },
        { label: "Excess on this line", value: over ? (v.billed - v.card) * v.hours : 0, unit: "USD" },
        { label: "Flagged", text: over ? "yes, recoverable" : "no, within tolerance", strong: true },
      ];
    },
  },
  {
    id: "duplicates",
    icon: "ph-copy",
    group: "Procurement",
    name: "Duplicate invoice lines",
    equation: "key = (vendor, bill, title, quantity, amount)      every copy after the first by line id is a duplicate",
    why: "A duplicate's whole amount is leakage, not just the excess, so it is never also counted as an overbill. Ordering by line id makes which copy is the original deterministic.",
    inputs: [
      { key: "copies", label: "Copies of the line", value: 3, step: 1, unit: "count", max: 10 },
      { key: "amount", label: "Amount on the line", value: 16000, step: 500, unit: "USD" },
    ],
    compute: (v) => {
      const dupes = Math.max(0, v.copies - 1);
      return [
        { label: "Duplicate copies", value: dupes, decimals: 0 },
        { label: "Leakage", value: dupes * v.amount, unit: "USD", strong: true },
      ];
    },
  },
  {
    id: "benchmarks",
    icon: "ph-chart-bar",
    group: "Procurement",
    name: "Rate benchmarks",
    equation: "p25, median, p75 of effective USD per baseline hour,   grouped by role and location tier",
    why: "The spread matters more than the average: a rate at the 75th percentile is not wrong, it just has twenty-five percent of the market underneath it, and that is the negotiating position.",
    inputs: [
      { key: "rate", label: "Your rate", value: 96, step: 1, unit: "USD" },
      { key: "p25", label: "p25 for this role and tier", value: 74, step: 1, unit: "USD" },
      { key: "median", label: "Median", value: 88, step: 1, unit: "USD" },
      { key: "hours", label: "Hours a year", value: 1800, step: 100, unit: "count" },
    ],
    compute: (v) => [
      { label: "Above the median by", value: v.rate - v.median, unit: "USD" },
      { label: "A year at your rate", value: v.rate * v.hours, unit: "USD" },
      { label: "A year at the median", value: v.median * v.hours, unit: "USD" },
      { label: "Moving to p25 would save", value: Math.max(0, v.rate - v.p25) * v.hours, unit: "USD", strong: true },
    ],
  },
  {
    id: "harmonise",
    icon: "ph-align-center-horizontal",
    group: "Procurement",
    name: "Rate harmonisation",
    equation: "harmonised = min( card_rate, median for the role and tier )",
    why: "Never a target above what you already pay. A vendor already under the median keeps its rate, so harmonisation can only move spend in one direction and the saving is never an artefact of the benchmark.",
    inputs: [
      { key: "card", label: "Card rate", value: 104, step: 1, unit: "USD" },
      { key: "median", label: "Median for the role and tier", value: 88, step: 1, unit: "USD" },
      { key: "hours", label: "Hours", value: 4200, step: 100, unit: "count" },
    ],
    compute: (v) => {
      const harmonised = Math.min(v.card, v.median);
      return [
        { label: "Harmonised rate", value: harmonised, unit: "USD" },
        { label: "Saving", value: (v.card - harmonised) * v.hours, unit: "USD", strong: true },
      ];
    },
  },
  {
    id: "fx",
    icon: "ph-currency-circle-dollar",
    group: "Procurement",
    name: "FX re-translation",
    equation: "fx_effect = card_rate × quantity × ( usd_per_unit(as_of) - usd_per_unit(posted) )",
    why: "Kept as its own step of the savings bridge so a currency move can never be presented as a negotiated saving. ADR 0003 is this one line.",
    inputs: [
      { key: "rate", label: "Card rate, local currency", value: 2500, step: 50, unit: "count" },
      { key: "hours", label: "Hours", value: 1600, step: 50, unit: "count" },
      { key: "posted", label: "USD per unit when posted", value: 0.0125, step: 0.0001, unit: "rate", max: 0.05 },
      { key: "asOf", label: "USD per unit as of now", value: 0.012, step: 0.0001, unit: "rate", max: 0.05 },
    ],
    compute: (v) => {
      const effect = v.rate * v.hours * (v.asOf - v.posted);
      return [
        { label: "Cost at the posted rate", value: v.rate * v.hours * v.posted, unit: "USD" },
        { label: "Cost at today's rate", value: v.rate * v.hours * v.asOf, unit: "USD" },
        { label: "Currency effect, not a saving", value: effect, unit: "USD", strong: true },
      ];
    },
  },
  {
    id: "savings-bridge",
    icon: "ph-steps",
    group: "Procurement",
    name: "The savings bridge",
    equation: "current = leakage + fx + harmonisation + volume_shift + exits + new",
    why: "Five ordered steps from what you pay now to what you would pay, in integer cents so they sum exactly. Every step traces to the invoice lines it moved, which is what makes a saving claim auditable.",
    inputs: [
      { key: "current", label: "Current cost", value: 24000000, step: 500000, unit: "USD" },
      { key: "leakage", label: "Leakage recovered", value: 1300000, step: 50000, unit: "USD" },
      { key: "fx", label: "FX re-translation", value: -240000, step: 20000, unit: "USD", min: -2000000 },
      { key: "harmonisation", label: "Rate harmonisation", value: 2100000, step: 50000, unit: "USD" },
      { key: "volume", label: "Volume shift", value: 1450000, step: 50000, unit: "USD" },
      { key: "exits", label: "Vendor exits", value: 980000, step: 50000, unit: "USD" },
    ],
    compute: (v) => {
      const saved = v.leakage + v.fx + v.harmonisation + v.volume + v.exits;
      return [
        { label: "Total movement", value: saved, unit: "USD" },
        { label: "New cost", value: v.current - saved, unit: "USD" },
        { label: "Saving against today", value: saved, unit: "USD", strong: true },
      ];
    },
  },
  {
    id: "tail",
    icon: "ph-handshake",
    group: "Procurement",
    name: "Tail vendor targets",
    equation: "target = min( current_rate, p25 )       walk_away = min( current_rate, median )",
    why: "For a vendor off the panel but worth keeping as a backup. Both bounds are capped at today's price, so a negotiation brief can never open by asking for a rate higher than the one in force.",
    inputs: [
      { key: "current", label: "Current rate", value: 92, step: 1, unit: "USD" },
      { key: "p25", label: "p25", value: 74, step: 1, unit: "USD" },
      { key: "median", label: "Median", value: 88, step: 1, unit: "USD" },
      { key: "quality", label: "Quality score", value: 0.72, step: 0.01, unit: "share", max: 1 },
      { key: "floor", label: "Quality floor", value: 0.6, step: 0.01, unit: "share", max: 1 },
    ],
    compute: (v) => {
      if (v.quality < v.floor) return [{ label: "Plan", text: "below the quality floor: exit", strong: true }];
      return [
        { label: "Target rate", value: Math.min(v.current, v.p25), unit: "USD" },
        { label: "Walk-away rate", value: Math.min(v.current, v.median), unit: "USD" },
        { label: "Plan", text: v.current <= v.median ? "negotiate and keep as a backup" : "exit", strong: true },
      ];
    },
  },
  {
    id: "verify",
    icon: "ph-seal-check",
    group: "Governance",
    name: "The answer check",
    equation: "answer is served  ⟺  every cited value == the value recorded against its lineage id",
    why: "The last gate before anything reaches a page. A figure in the text is compared with the figure in the lineage record, and a mismatch refuses the whole answer rather than correcting it quietly.",
    inputs: [
      { key: "cited", label: "Value in the answer", value: 63814, step: 1, unit: "USD" },
      { key: "recorded", label: "Value recorded against the lineage id", value: 63814, step: 1, unit: "USD" },
    ],
    compute: (v) => [
      { label: "Difference", value: v.cited - v.recorded, unit: "USD" },
      { label: "Outcome", text: v.cited === v.recorded ? "served" : "refused, the answer is blocked", strong: true },
    ],
  },
  {
    id: "run-cost",
    icon: "ph-coins",
    group: "Governance",
    name: "What a run costs",
    equation: "cost = Σ ( input_tokens × input_price + output_tokens × output_price ) per million",
    why: "Recorded per call in the audit schema and shown in the rail, because an analytics agent whose cost nobody can see is one nobody can budget for.",
    inputs: [
      { key: "calls", label: "Calls in the run", value: 24, step: 1, unit: "count", max: 400 },
      { key: "inTok", label: "Input tokens per call", value: 4200, step: 100, unit: "count" },
      { key: "outTok", label: "Output tokens per call", value: 900, step: 50, unit: "count" },
      { key: "inPrice", label: "Input price per million", value: 3, step: 0.25, unit: "USD", max: 30 },
      { key: "outPrice", label: "Output price per million", value: 15, step: 0.25, unit: "USD", max: 100 },
    ],
    compute: (v) => {
      const cost = (v.calls * (v.inTok * v.inPrice + v.outTok * v.outPrice)) / 1_000_000;
      return [
        { label: "Tokens in", value: v.calls * v.inTok, decimals: 0 },
        { label: "Tokens out", value: v.calls * v.outTok, decimals: 0 },
        { label: "Run cost", value: cost, unit: "USD", decimals: 4, strong: true },
      ];
    },
  },
];

/* ---------- Rendering ---------- */

export async function render(view) {
  view.replaceChildren(head(), jumpBar());
  for (const [group, icon] of GROUPS) {
    const models = MODELS.filter((m) => m.group === group);
    const heading = el("h2", { id: groupId(group) }, [
      el("i", { class: `ph-light ${icon} group-icon`, "aria-hidden": "true" }),
      group,
    ]);
    const node = el("section", { class: "section" }, [heading, ...models.map((model, i) => panel(model, i))]);
    view.append(node);
  }
  view.append(
    el("p", {
      class: "note note--advice",
      text: "These panels recompute in your browser from the inputs you set, so you can see how each equation behaves. The figures on the pack pages are computed by the same equations on the server, where each one carries a lineage id back to its rows.",
    }),
  );
}

const groupId = (group) => `group-${group.toLowerCase().replace(/[^a-z]+/g, "-")}`;

/** Twenty-eight panels is a long page, so the groups get a bar that jumps to them.

    These are buttons, not anchors. The workspace routes on the hash, so an `href="#group-hr"` would
    read as a page name, fail to match one and send the router back to FP&A. */
function jumpBar() {
  return el(
    "nav",
    { class: "jump", "aria-label": "Formula groups" },
    GROUPS.map(([group, icon]) =>
      el(
        "button",
        {
          type: "button",
          class: "jump-link",
          onclick: () =>
            document.getElementById(groupId(group))?.scrollIntoView({ behavior: "smooth", block: "start" }),
        },
        [
          el("i", { class: `ph-light ${icon}`, "aria-hidden": "true" }),
          el("span", { text: group }),
          el("span", { class: "jump-n", text: String(MODELS.filter((m) => m.group === group).length) }),
        ],
      ),
    ),
  );
}

function head() {
  return el("div", { class: "page-head" }, [
    el("h1", { text: "The arithmetic" }),
    el("p", {
      text: "Every number this platform shows comes from one of these. Move an input and watch the result change, then go to the pack page and click the same figure to see the rows it came from.",
    }),
  ]);
}

function panel(model, index) {
  const state = Object.fromEntries(model.inputs.map((i) => [i.key, i.value]));
  const results = el("dl", { class: "formula-out" });

  const recompute = () => {
    const rows = model.compute(state);
    results.replaceChildren(
      ...rows.flatMap((row) => {
        const value =
          row.text !== undefined
            ? row.text
            : row.value === null || row.value === undefined || Number.isNaN(row.value)
              ? "not defined here"
              : row.decimals !== undefined
                ? fixed(row)
                : fmt(row.value, row.unit);
        const dt = el("dt", { class: row.strong ? "is-strong" : null, text: row.label });
        const dd = el("dd", { class: row.strong ? "is-strong" : null, text: value ?? "" });
        return [dt, dd];
      }),
    );
  };

  const controls = model.inputs.map((input) => {
    const id = `${model.id}-${input.key}`;
    const output = el("output", { for: id, class: "field-value", text: display(state[input.key], input) });
    const slider = el("input", {
      id,
      type: "range",
      min: String(input.min ?? 0),
      max: String(input.max ?? (Math.abs(input.value) * 2.5 || 1)),
      step: String(input.step),
      value: String(input.value),
      oninput: (event) => {
        state[input.key] = Number(event.target.value);
        output.textContent = display(state[input.key], input);
        recompute();
      },
    });
    return el("div", { class: "field field--slider" }, [
      el("label", { for: id, text: input.label }),
      slider,
      output,
    ]);
  });

  recompute();

  return card(
    [
      el("h3", { class: "formula-name" }, [
        el("i", { class: `ph-light ${model.icon} formula-icon`, "aria-hidden": "true" }),
        model.name,
      ]),
      el("code", { class: "formula", text: model.equation }),
      el("p", { class: "kpi-note", text: model.why }),
      el("div", { class: "formula-body" }, [el("div", { class: "formula-inputs" }, controls), results]),
    ],
    { delay: index * 60 },
  );
}

/** A result that names its own decimals: money keeps its symbol, a percentage keeps its sign. */
function fixed(row) {
  const digits = { minimumFractionDigits: row.decimals, maximumFractionDigits: row.decimals };
  if (row.unit === "USD") return row.value.toLocaleString("en-US", { style: "currency", currency: "USD", ...digits });
  if (row.unit === "pct") return `${row.value.toLocaleString("en-US", digits)}%`;
  return row.value.toLocaleString("en-US", digits);
}

function display(value, input) {
  if (input.unit === "share" || input.unit === "rate") return `${(value * 100).toFixed(0)}%`;
  if (input.unit === "ratio") return value.toFixed(2);
  return fmt(value, input.unit);
}
