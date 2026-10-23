select instalment_id, policy_id, lob, method, due_date, amount, paid_date
from {{ source('raw_fpa', 'instalments') }}
