select
    line_id,
    tranid,
    trandate,
    month,
    entity as vendor_id,
    subsidiary,
    currency,
    exchangerate,
    billed_title,
    location_tier,
    pricing_model,
    quantity,
    unit,
    reported_hours,
    rate,
    foreignamount,
    amount
from {{ source('raw_ratecards', 'invoice_lines') }}
