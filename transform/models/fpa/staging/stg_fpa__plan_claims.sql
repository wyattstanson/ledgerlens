select month, lob, channel, approved_count, approved_amount, payouts, paid_amount, lae_amount
from {{ source('raw_fpa', 'plan_claims') }}
