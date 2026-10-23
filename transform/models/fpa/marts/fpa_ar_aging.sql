{{ config(materialized='table') }}
-- Instalments due on or before the as-of date and not yet paid by it.
with as_of as (select max(enddate) as as_of from {{ ref('stg_fpa__accountingperiod') }})
select
    i.instalment_id,
    i.lob,
    i.method,
    i.due_date,
    i.amount,
    date_diff('day', i.due_date, a.as_of) as days_past_due,
    a.as_of
from {{ ref('stg_fpa__instalments') }} i
cross join as_of a
where i.due_date <= a.as_of and (i.paid_date is null or i.paid_date > a.as_of)
