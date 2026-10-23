select id, "transaction", account, amount, foreignamount, department, "class", location
from {{ source('raw_fpa', 'transactionline') }}
