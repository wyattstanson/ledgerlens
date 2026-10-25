"""Read-only payments feed shaped like a carrier payments network (BUILD_PROMPT 10.2).

For the insurer: premium received, claim paid and payout returned, each with its instrument (payment method
or payout channel) and timestamp. For the processor (``--variant processor``): network settlements,
chargebacks debited and chargebacks recovered from merchants, with the card type as the instrument.
Every event carries the feed's credit/debit indicator (``direction``: in or out, from the account
holder's side).

``PaymentsFeed`` is the seam: the tools get their feed from ``payments_feed()``, and every method returns
``Rows`` with its own source, so a real feed (for example a carrier network's MCP server) can replace
``DuckDBPaymentsFeed`` without touching the tools or their lineage. There are no write methods.
"""

from datetime import date
from pathlib import Path
from typing import Literal, Protocol

from ledgerlens.core.db import NO_EXTERNAL_ACCESS, connect, fetch_dicts
from ledgerlens.fpa.source import Rows

EventKind = Literal[
    "premium_received", "claim_paid", "payout_returned", "settlement", "chargeback", "chargeback_recovered"
]
MAX_EVENTS = 200
SOURCE = "fpa.stg_fpa__payment_events"

EVENTS_SQL = """
select event_id, kind, event_ts, amount, direction, instrument, lob, ref_id
from fpa.stg_fpa__payment_events
where (?::varchar is null or kind = ?)
  and (?::date is null or event_date >= ?)
  and (?::date is null or event_date <= ?)
order by event_ts desc, event_id desc
limit ?
"""

SUMMARY_SQL = """
select kind, instrument, direction, count(*) as events, sum(amount) as amount,
       list(event_id order by event_id) as event_ids
from fpa.stg_fpa__payment_events
where strftime(event_date, '%Y-%m') = ?
group by all
order by kind, instrument, direction
"""


class PaymentsFeed(Protocol):
    def events(self, *, kind: EventKind | None, start: date | None, end: date | None, limit: int) -> Rows: ...

    def summary(self, month: str) -> Rows: ...


class DuckDBPaymentsFeed:
    """The synthetic feed in the warehouse."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def events(
        self,
        *,
        kind: EventKind | None = None,
        start: date | None = None,
        end: date | None = None,
        limit: int = 50,
    ) -> Rows:
        """The most recent events first, at most ``MAX_EVENTS``."""
        cap = max(1, min(limit, MAX_EVENTS))
        with connect(self.db_path, read_only=True, config=NO_EXTERNAL_ACCESS) as conn:
            rows = fetch_dicts(conn, EVENTS_SQL, [kind, kind, start, start, end, end, cap])
        return Rows(rows, EVENTS_SQL, [SOURCE])

    def summary(self, month: str) -> Rows:
        with connect(self.db_path, read_only=True, config=NO_EXTERNAL_ACCESS) as conn:
            rows = fetch_dicts(conn, SUMMARY_SQL, [month])
        return Rows(rows, SUMMARY_SQL, [SOURCE])


def payments_feed(db_path: Path) -> PaymentsFeed:
    """The feed the tools read. Swap the class here to point them at a different feed."""
    return DuckDBPaymentsFeed(db_path)
