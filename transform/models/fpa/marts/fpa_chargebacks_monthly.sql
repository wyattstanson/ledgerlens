{{ config(materialized='table') }}
-- Processor chargebacks per vertical and initiation. Event basis, by the month raised: how many, the amount
-- disputed, and the net amount the merchant has not repaid. Cash basis: debited by the networks (when raised)
-- and recovered from merchants (when repaid). Write-offs by the month written off. transactions are the
-- approved purchases captured that month, the denominator of the chargeback rate as the networks count it.
-- row_key: month|lob|initiation. Empty when the warehouse holds the insurer.
with captured as (
    select strftime(capture_date, '%Y-%m') as month, lob, initiation, sum(approvals)::bigint as transactions
    from {{ ref('stg_fpa__captures') }}
    group by all
),
raised as (
    select strftime(raised_date, '%Y-%m') as month, lob, initiation,
           count(*) as chargebacks, sum(amount) as disputed_amount,
           coalesce(sum(amount) filter (where recovered_date is null), 0) as net_amount
    from {{ ref('stg_fpa__chargebacks') }}
    group by all
),
recovered as (
    select strftime(recovered_date, '%Y-%m') as month, lob, initiation, sum(amount) as recovered_amount
    from {{ ref('stg_fpa__chargebacks') }}
    where recovered_date is not null
    group by all
),
written_off as (
    select strftime(written_off_date, '%Y-%m') as month, lob, initiation, sum(amount) as written_off_amount
    from {{ ref('stg_fpa__chargebacks') }}
    where written_off_date is not null
    group by all
),
keys as (
    select month, lob, initiation from captured
    union select month, lob, initiation from raised
    union select month, lob, initiation from recovered
    union select month, lob, initiation from written_off
),
months as (select strftime(startdate, '%Y-%m') as month from {{ ref('stg_fpa__accountingperiod') }})
select
    k.month || '|' || k.lob || '|' || k.initiation as row_key,
    k.month, k.lob, k.initiation,
    coalesce(t.transactions, 0) as transactions,
    coalesce(r.chargebacks, 0) as chargebacks,
    coalesce(r.disputed_amount, 0) as disputed_amount,
    coalesce(r.net_amount, 0) as net_amount,
    coalesce(r.disputed_amount, 0) as debited_amount,
    coalesce(c.recovered_amount, 0) as recovered_amount,
    coalesce(w.written_off_amount, 0) as written_off_amount
from keys k
join months using (month)
left join captured t using (month, lob, initiation)
left join raised r using (month, lob, initiation)
left join recovered c using (month, lob, initiation)
left join written_off w using (month, lob, initiation)
