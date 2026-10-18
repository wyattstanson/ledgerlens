-- Warn (not fail): the synthetic world plants duplicate lines on purpose; leakage recovery reclaims them.
{{ config(severity='warn') }}
select vendor_id, tranid, billed_title, quantity, foreignamount, count(*) as copies
from {{ ref('stg_rc__invoice_lines') }}
group by all
having count(*) > 1
