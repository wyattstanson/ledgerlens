{{ config(materialized='table') }}
-- Billing by due month and collections by paid month, per line and payment method. row_key: month|lob|method.
-- Cells come from both sides, so a month with collections but nothing due (late payers in a sparse cell)
-- keeps its month and key.
with due as (
    select strftime(due_date, '%Y-%m') as month, lob, method,
           count(*) as instalments_due, sum(amount) as due_amount,
           count(*) filter (where paid_date is not null and paid_date <= due_date) as paid_on_time
    from {{ ref('stg_fpa__instalments') }}
    group by all
),
collected as (
    select strftime(e.event_date, '%Y-%m') as month, e.lob, e.instrument as method,
           count(*) as payments, sum(e.amount) as collected_amount
    from {{ ref('stg_fpa__payment_events') }} e
    where e.kind = 'premium_received'
    group by all
),
months as (select strftime(startdate, '%Y-%m') as month from {{ ref('stg_fpa__accountingperiod') }}),
cells as (
    select month, lob, method from due
    union
    select month, lob, method from collected
)
select
    c.month || '|' || c.lob || '|' || c.method as row_key,
    c.month,
    c.lob,
    c.method,
    coalesce(d.instalments_due, 0) as instalments_due,
    coalesce(d.due_amount, 0) as due_amount,
    coalesce(d.paid_on_time, 0) as paid_on_time,
    coalesce(k.payments, 0) as payments,
    coalesce(k.collected_amount, 0) as collected_amount
from cells c
join months m on m.month = c.month
left join due d on d.month = c.month and d.lob = c.lob and d.method = c.method
left join collected k on k.month = c.month and k.lob = c.lob and k.method = c.method
