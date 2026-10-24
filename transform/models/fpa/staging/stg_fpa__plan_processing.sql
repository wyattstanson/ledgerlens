select month, lob, card_type, initiation, attempts, transactions, payment_volume, fees_captured, interchange_captured, fees_settled, interchange_settled
from {{ source('raw_fpa', 'plan_processing') }}
