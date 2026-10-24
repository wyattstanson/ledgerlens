-- Processor: every network settlement in the payments feed is its batch's captured fees net of interchange, on
-- the batch's settlement date, once; and every batch with approvals that settles inside the window has one.
-- Returns batches that break this. Passes trivially when the warehouse holds the insurer.
with batches as (
    select batch_id, any_value(settle_date) as settle_date, sum(fee_amount) - sum(interchange_amount) as net
    from {{ ref('stg_fpa__captures') }}
    group by batch_id
    having sum(approvals) > 0
),
window_ as (
    select min(startdate) as first_day, max(enddate) as last_day from {{ ref('stg_fpa__accountingperiod') }}
),
expected as (
    select b.batch_id, b.settle_date, b.net
    from batches b, window_ w
    where b.settle_date between w.first_day and w.last_day
),
feed as (
    select ref_id as batch_id, event_date, case direction when 'out' then -amount else amount end as net,
           count(*) over (partition by ref_id) as events
    from {{ ref('stg_fpa__payment_events') }}
    where kind = 'settlement'
)
select coalesce(e.batch_id, f.batch_id) as batch_id, e.net as captured_net, f.net as settled_net
from expected e full join feed f using (batch_id)
where e.batch_id is null or f.batch_id is null or f.events <> 1 or f.event_date <> e.settle_date
   or abs(e.net - f.net) > 0.005
