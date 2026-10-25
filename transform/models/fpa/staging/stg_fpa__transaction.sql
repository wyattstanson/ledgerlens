select id, tranid, trandate, recordtype, entity, subsidiary, postingperiod, posting, approvalstatus, currency, exchangerate, amount, foreignamount, memo, custbody_event_id
from {{ source('raw_fpa', 'transaction') }}
