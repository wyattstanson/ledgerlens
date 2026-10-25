{{ config(materialized='table') }}
-- Approvals by the claim's channel; payouts, returns and handling cost by the paying leg's channel.
-- row_key: month|lob|channel.
with approved as (
    select strftime(approved_date, '%Y-%m') as month, lob, channel,
           count(*) as approved_count, sum(severity) as approved_amount
    from {{ ref('stg_fpa__claims') }}
    group by all
),
paid as (
    select strftime(paid_ts, '%Y-%m') as month, lob, channel,
           count(*) as payouts, sum(amount) as paid_amount, sum(handling_cost) as lae_amount,
           avg(date_diff('hour', approved_date::timestamp, paid_ts) / 24.0) as days_to_pay
    from {{ ref('stg_fpa__payouts') }}
    where paid_ts is not null
    group by all
),
returned as (
    select strftime(returned_ts, '%Y-%m') as month, lob, channel, sum(amount) as returned_amount
    from {{ ref('stg_fpa__payouts') }}
    where returned_ts is not null
    group by all
),
keys as (
    select month, lob, channel from approved
    union select month, lob, channel from paid
    union select month, lob, channel from returned
),
months as (select strftime(startdate, '%Y-%m') as month from {{ ref('stg_fpa__accountingperiod') }})
select
    k.month || '|' || k.lob || '|' || k.channel as row_key,
    k.month, k.lob, k.channel,
    coalesce(a.approved_count, 0) as approved_count,
    coalesce(a.approved_amount, 0) as approved_amount,
    coalesce(p.payouts, 0) as payouts,
    coalesce(p.paid_amount, 0) as paid_amount,
    coalesce(r.returned_amount, 0) as returned_amount,
    coalesce(p.lae_amount, 0) as lae_amount,
    p.days_to_pay
from keys k
join months using (month)
left join approved a using (month, lob, channel)
left join paid p using (month, lob, channel)
left join returned r using (month, lob, channel)
