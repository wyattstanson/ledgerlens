select claim_id, policy_id, lob, channel, occurred_date, approved_date, severity
from {{ source('raw_fpa', 'claims') }}
