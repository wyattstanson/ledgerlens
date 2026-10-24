select month, lob, initiation, chargebacks, disputed_amount, net_amount, debited_amount, recovered_amount
from {{ source('raw_fpa', 'plan_chargebacks') }}
