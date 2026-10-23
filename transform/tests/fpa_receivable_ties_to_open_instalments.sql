-- The receivable account at the latest period end equals the instalments billed or carried in and still unpaid.
with gl as (
    select sum(l.amount) as balance
    from {{ ref('stg_fpa__transactionline') }} l
    where l.account = 1200
),
open_items as (
    select sum(amount) as balance from {{ ref('fpa_ar_aging') }}
)
select gl.balance as gl_balance, open_items.balance as open_balance
from gl, open_items
where abs(coalesce(gl.balance, 0) - coalesce(open_items.balance, 0)) > 0.005
