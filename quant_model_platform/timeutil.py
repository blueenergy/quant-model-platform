"""UTC datetime convention for everything this platform stores.

Every timestamp written to Mongo is UTC. BSON dates carry no timezone, and
pymongo decodes them as *naive* unless the client asks otherwise, so a stored
value compared against the host's wall clock is silently off by the host's UTC
offset (8 hours here). The rule is therefore:

* **write** with :func:`utcnow` — timezone-aware UTC;
* **read** through a tz-aware client (``MongoClient(..., tz_aware=True)``), so
  decoded values are aware too;
* **display** with :func:`to_cst`, which also accepts the naive values written
  by older code and read from the live database.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

#: China Standard Time. Fixed +08:00 — no DST — so avoid a tzdata dependency.
CST = timezone(timedelta(hours=8), "CST")


def utcnow() -> datetime:
    """Current time as an aware UTC datetime; use for every DB write."""
    return datetime.now(timezone.utc)


def to_cst(value: datetime) -> datetime:
    """Render a stored timestamp in CST for logs, reports and comparisons.

    Naive input is read as UTC, which is what the database actually holds.
    """
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(CST)
