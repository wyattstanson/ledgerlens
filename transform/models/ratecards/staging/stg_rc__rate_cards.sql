select
    card_id,
    vendor_id,
    vendor_title,
    location_tier,
    currency,
    pricing_model,
    rate,
    productivity_commitment,
    contract_start
from {{ source('raw_ratecards', 'rate_cards') }}
