from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


TRACE_PAGE_SIZE = 1000


def collect_trace_pages(
    fetch_page: Callable[[dict[str, Any]], Mapping[str, Any]],
    records_key: str,
    *,
    params: Mapping[str, Any] | None = None,
    page_size: int = TRACE_PAGE_SIZE,
) -> list[dict[str, Any]]:
    records = []
    cursor = None
    seen_cursors = set()
    page_size = max(page_size, 1)
    while True:
        query = dict(params or {})
        query["limit"] = page_size
        if cursor is not None:
            query["cursor"] = cursor
        payload = fetch_page(query)
        if not isinstance(payload, Mapping):
            raise ValueError("trace page response is not an object")
        page = payload.get(records_key)
        if not isinstance(page, list):
            raise ValueError(f"trace response field {records_key!r} is not a list")
        records.extend(page)

        next_cursor = payload.get("next_cursor")
        if next_cursor is None:
            if payload.get("truncated") is True:
                raise ValueError(
                    "trace response is truncated without a continuation cursor"
                )
            return records
        if not isinstance(next_cursor, str) or next_cursor in seen_cursors:
            raise ValueError(f"invalid trace continuation cursor: {next_cursor!r}")
        seen_cursors.add(next_cursor)
        cursor = next_cursor
