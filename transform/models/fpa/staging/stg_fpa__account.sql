select id, acctnumber, fullname, accttype, parent
from {{ source('raw_fpa', 'account') }}
