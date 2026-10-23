select month, amount
from {{ source('raw_fpa', 'opex') }}
