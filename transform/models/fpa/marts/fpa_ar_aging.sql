{{ config(materialized='table') }}
-- Receivables open at the latest period end, with days past due. Insurer: premium instalments due and not yet
-- paid. Processor: settlement batches captured and not yet settled (due on their settlement date, so not past
-- due yet), and chargebacks the merchant has neither repaid nor had written off (due when raised).
with as_of as (select max(enddate) as as_of from {{ ref('stg_fpa__accountingperiod') }}),
items as (
    select i.instalment_id as item_id, 'premium_instalment' as item_type, i.lob, i.method as instrument,
           i.due_date, i.amount
    from {{ ref('stg_fpa__instalments') }} i
    cross join as_of a
    where i.due_date <= a.as_of and (i.paid_date is null or i.paid_date > a.as_of)
    union all
    select c.batch_id, 'settlement_batch', any_value(c.lob), any_value(c.card_type), any_value(c.settle_date),
           sum(c.fee_amount) - sum(c.interchange_amount)
    from {{ ref('stg_fpa__captures') }} c
    cross join as_of a
    where c.capture_date <= a.as_of and c.settle_date > a.as_of
    group by c.batch_id
    having sum(c.approvals) > 0
    union all
    select k.chargeback_id, 'merchant_chargeback', k.lob, k.card_type, k.raised_date, k.amount
    from {{ ref('stg_fpa__chargebacks') }} k
    cross join as_of a
    where k.raised_date <= a.as_of
      and (k.recovered_date is null or k.recovered_date > a.as_of)
      and (k.written_off_date is null or k.written_off_date > a.as_of)
)
select
    i.item_id,
    i.item_type,
    i.lob,
    i.instrument,
    i.due_date,
    i.amount,
    date_diff('day', i.due_date, a.as_of) as days_past_due,
    a.as_of
from items i
cross join as_of a
