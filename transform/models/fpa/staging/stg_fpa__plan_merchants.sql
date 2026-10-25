select month, lob, active_merchants, new_merchants
from {{ source('raw_fpa', 'plan_merchants') }}
