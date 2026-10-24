-- Processor: every chargeback debit and merchant recovery in the payments feed matches a dispute record (same
-- amount, merchant and date, once), and every dispute raised or recovered inside the window has its event.
-- Returns the mismatches. Passes trivially when the warehouse holds the insurer.
with window_ as (
    select min(startdate) as first_day, max(enddate) as last_day from {{ ref('stg_fpa__accountingperiod') }}
),
expected as (
    select 'chargeback' as kind, d.chargeback_id, d.merchant_id, d.raised_date as event_date, d.amount
    from {{ ref('stg_fpa__chargebacks') }} d, window_ w
    where d.raised_date between w.first_day and w.last_day
    union all
    select 'chargeback_recovered', d.chargeback_id, d.merchant_id, d.recovered_date, d.amount
    from {{ ref('stg_fpa__chargebacks') }} d, window_ w
    where d.recovered_date between w.first_day and w.last_day
),
feed as (
    select kind, ref_id as chargeback_id, merchant_id, event_date, amount,
           count(*) over (partition by kind, ref_id) as events
    from {{ ref('stg_fpa__payment_events') }}
    where kind in ('chargeback', 'chargeback_recovered')
)
select coalesce(e.kind, f.kind) as kind, coalesce(e.chargeback_id, f.chargeback_id) as chargeback_id,
       e.amount as dispute_amount, f.amount as event_amount
from expected e full join feed f using (kind, chargeback_id)
where e.chargeback_id is null or f.chargeback_id is null or f.events <> 1
   or f.event_date <> e.event_date or f.merchant_id <> e.merchant_id or abs(e.amount - f.amount) > 0.005
