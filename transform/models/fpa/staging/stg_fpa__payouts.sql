select payout_id, claim_id, lob, leg, channel, amount, handling_cost, approved_date, paid_ts, returned_ts
from {{ source('raw_fpa', 'payouts') }}
