select event_id, kind, event_ts, event_date, amount, direction, instrument, lob, policy_id, claim_id, merchant_id, ref_id
from {{ source('raw_fpa', 'payment_events') }}
