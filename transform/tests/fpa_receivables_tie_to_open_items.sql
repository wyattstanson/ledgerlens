-- Each receivable account at the latest period end equals the open items behind it: premiums receivable (1200)
-- the instalments billed or carried in and still unpaid; the settlement receivable (1100) the batches captured
-- and not yet settled; merchant receivables (1300) the chargebacks neither recovered nor written off.
with gl as (
    select l.account, sum(l.amount) as balance
    from {{ ref('stg_fpa__transactionline') }} l
    join {{ ref('stg_fpa__account') }} a on a.id = l.account
    where a.accttype = 'AcctRec'
    group by all
),
open_items as (
    select case item_type
               when 'premium_instalment' then 1200
               when 'settlement_batch' then 1100
               when 'merchant_chargeback' then 1300
           end as account,
           sum(amount) as balance
    from {{ ref('fpa_ar_aging') }}
    group by all
)
select coalesce(gl.account, open_items.account) as account, gl.balance as gl_balance,
       open_items.balance as open_balance
from gl full join open_items using (account)
where abs(coalesce(gl.balance, 0) - coalesce(open_items.balance, 0)) > 0.005
