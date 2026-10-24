select chargeback_id, merchant_id, lob, card_type, initiation, purchase_date, raised_date, amount, recovered_date, written_off_date
from {{ source('raw_fpa', 'chargebacks') }}
