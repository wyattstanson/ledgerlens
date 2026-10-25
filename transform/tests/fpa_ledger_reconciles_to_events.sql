-- BUILD_PROMPT 10.1: the ledger reconciles exactly to the payments feed every month. Returns months where
-- the cash postings that cite an event differ from the events themselves, or where an event has no posting.
with gl as (
    select strftime(t.trandate, '%Y-%m') as month, sum(l.amount) as gl_cash, count(distinct t.id) as postings
    from {{ ref('stg_fpa__transaction') }} t
    join {{ ref('stg_fpa__transactionline') }} l on l."transaction" = t.id and l.account = 1000
    where t.custbody_event_id is not null
    group by all
),
feed as (
    select strftime(event_date, '%Y-%m') as month,
           sum(case kind when 'claim_paid' then -amount else amount end) as event_cash,
           count(*) as events
    from {{ ref('stg_fpa__payment_events') }}
    group by all
)
select coalesce(gl.month, feed.month) as month, gl.gl_cash, feed.event_cash, gl.postings, feed.events
from gl full join feed using (month)
where gl.gl_cash is null or feed.event_cash is null
   or abs(gl.gl_cash - feed.event_cash) > 0.005 or gl.postings <> feed.events
