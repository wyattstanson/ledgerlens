{{ config(materialized='table') }}
-- Cash account (1000) movements per period, split by what caused them. Positive is cash in.
select
    p.periodname,
    strftime(p.startdate, '%Y-%m') as month,
    p.closed,
    sum(l.amount) filter (where t.memo = 'premium received') as collected_premium,
    -sum(l.amount) filter (where t.memo = 'claim payout') as claims_paid,
    sum(l.amount) filter (where t.memo = 'payout returned') as payouts_returned,
    -sum(l.amount) filter (where t.memo = 'claim handling cost') as lae,
    -sum(l.amount) filter (where t.memo = 'operating expense') as opex,
    sum(l.amount) as net_operating_cash,
    list(t.id order by t.id) filter (where t.recordtype = 'journalentry' and t.memo = 'operating expense')
        as opex_transaction_ids
from {{ ref('stg_fpa__transactionline') }} l
join {{ ref('stg_fpa__transaction') }} t on t.id = l."transaction"
join {{ ref('stg_fpa__accountingperiod') }} p on p.id = t.postingperiod
where l.account = 1000
group by all
