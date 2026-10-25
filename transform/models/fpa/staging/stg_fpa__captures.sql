select batch_id, capture_date, settle_date, lob, card_type, initiation, attempts, approvals, volume, fee_amount, interchange_amount
from {{ source('raw_fpa', 'captures') }}
