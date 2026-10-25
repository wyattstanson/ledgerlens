select month, lob, method, instalments_due, due_amount, collected_amount
from {{ source('raw_fpa', 'plan_premium') }}
