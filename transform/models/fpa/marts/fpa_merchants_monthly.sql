{{ config(materialized='table') }}
-- Processor merchants per vertical: active at the period end, onboarded and churned during it.
-- row_key: month|lob. Empty when the warehouse holds the insurer.
with periods as (
    select strftime(startdate, '%Y-%m') as month, startdate, enddate
    from {{ ref('stg_fpa__accountingperiod') }}
)
select
    p.month || '|' || m.lob as row_key,
    p.month,
    m.lob,
    count(*) filter (
        where m.onboard_date <= p.enddate and (m.churn_date is null or m.churn_date > p.enddate)
    ) as active_merchants,
    count(*) filter (where m.onboard_date between p.startdate and p.enddate) as new_merchants,
    count(*) filter (where m.churn_date between p.startdate and p.enddate) as churned_merchants
from periods p
cross join {{ ref('stg_fpa__merchants') }} m
group by all
