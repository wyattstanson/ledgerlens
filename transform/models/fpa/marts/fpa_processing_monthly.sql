{{ config(materialized='table') }}
-- Processor captures by capture month (event basis: attempts, approved transactions, volume, fees and
-- interchange) and by settlement month (cash basis), per vertical, card type and initiation.
-- row_key: month|lob|card_type|initiation. Empty when the warehouse holds the insurer.
with captured as (
    select strftime(capture_date, '%Y-%m') as month, lob, card_type, initiation,
           sum(attempts)::bigint as attempts, sum(approvals)::bigint as transactions,
           sum(volume) as payment_volume, sum(fee_amount) as fees_captured,
           sum(interchange_amount) as interchange_captured
    from {{ ref('stg_fpa__captures') }}
    group by all
),
settled as (
    select strftime(settle_date, '%Y-%m') as month, lob, card_type, initiation,
           sum(fee_amount) as fees_settled, sum(interchange_amount) as interchange_settled
    from {{ ref('stg_fpa__captures') }}
    group by all
),
cells as (
    select month, lob, card_type, initiation from captured
    union
    select month, lob, card_type, initiation from settled
),
months as (select strftime(startdate, '%Y-%m') as month from {{ ref('stg_fpa__accountingperiod') }})
select
    c.month || '|' || c.lob || '|' || c.card_type || '|' || c.initiation as row_key,
    c.month, c.lob, c.card_type, c.initiation,
    coalesce(k.attempts, 0) as attempts,
    coalesce(k.transactions, 0) as transactions,
    coalesce(k.payment_volume, 0) as payment_volume,
    coalesce(k.fees_captured, 0) as fees_captured,
    coalesce(k.interchange_captured, 0) as interchange_captured,
    coalesce(s.fees_settled, 0) as fees_settled,
    coalesce(s.interchange_settled, 0) as interchange_settled
from cells c
join months using (month)
left join captured k using (month, lob, card_type, initiation)
left join settled s using (month, lob, card_type, initiation)
