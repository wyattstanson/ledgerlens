select vendor_id, name, location_tier, currency, quality, sla
from {{ source('raw_ratecards', 'vendors') }}
