{{ config(materialized='table') }}
-- Weeks start on Monday. Claims are net of returned payouts.
select
    date_trunc('week', event_date) as week_start,
    sum(amount) filter (where kind = 'premium_received') as collected_premium,
    coalesce(sum(amount) filter (where kind = 'claim_paid'), 0)
        - coalesce(sum(amount) filter (where kind = 'payout_returned'), 0) as claims_paid,
    count(*) as events
from {{ ref('stg_fpa__payment_events') }}
group by all
