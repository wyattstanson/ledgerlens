{{ config(materialized='table') }}
-- Sum of posted lines per period and account (debits positive). row_key: month|account.
select
    strftime(p.startdate, '%Y-%m') || '|' || l.account as row_key,
    strftime(p.startdate, '%Y-%m') as month,
    p.periodname,
    p.closed,
    l.account,
    a.fullname,
    a.accttype,
    sum(l.amount) as amount,
    count(*) as lines
from {{ ref('stg_fpa__transactionline') }} l
join {{ ref('stg_fpa__transaction') }} t on t.id = l."transaction"
join {{ ref('stg_fpa__accountingperiod') }} p on p.id = t.postingperiod
join {{ ref('stg_fpa__account') }} a on a.id = l.account
where t.posting
group by all
