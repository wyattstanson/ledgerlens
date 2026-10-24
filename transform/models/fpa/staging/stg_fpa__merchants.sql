select merchant_id, lob, onboard_date, churn_date, attempts_per_day
from {{ source('raw_fpa', 'merchants') }}
