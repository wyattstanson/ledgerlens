select month, amount
from {{ source('raw_fpa', 'plan_opex') }}
