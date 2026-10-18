-- USD amount must equal the foreign amount at the exchange rate stamped on the line (ADR 0001),
-- within half a cent of rounding. Returns offending lines.
select line_id, amount, foreignamount, exchangerate
from {{ ref('stg_rc__invoice_lines') }}
where abs(amount - foreignamount * exchangerate) > 0.0051
