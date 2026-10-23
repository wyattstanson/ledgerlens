-- ADR 0001: every transaction balances. Returns transactions whose lines do not sum to zero.
select "transaction", sum(amount) as imbalance
from {{ ref('stg_fpa__transactionline') }}
group by "transaction"
having abs(sum(amount)) > 0.005
