select id, periodname, startdate, enddate, closed
from {{ source('raw_fpa', 'accountingperiod') }}
