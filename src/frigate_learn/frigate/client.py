"""Frigate 0.18 HTTP client.

The ONLY transport layer in the codebase. Everything Frigate-specific
(URLs, auth, pagination strategy, response shapes, retry policy) is contained
here and in this package's sibling modules.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterable, Sequence

import httpx

from .events import Event, parse_event
from .motion import MotionBucket, parse_motion_activity
from .reviews import Review, parse_review
from .snapshots import (
    annotated_region_crop_params,
    annotated_snapshot_params,
    annotated_snapshot_url,
    clean_snapshot_params,
    clean_snapshot_url,
    motion_activity_url,
    region_crop_params,
    review_preview_url,
)

logger = logging.getLogger(__name__)
debug = logger.debug

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class FrigateAPIError(Exception):
    """Raised for any non-2xx Frigate API response or transport failure."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        url: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.url = url
        self.retryable = retryable

    def __str__(self) -> str:
        bits = [self.message]
        if self.status_code is not None:
            bits.append(f"(status={self.status_code})")
        if self.url:
            bits.append(f"url={self.url}")
        return " ".join(bits)


class FrigateClient:
    """Thin HTTP wrapper around the Frigate 0.18 API."""

    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        retry_backoff: float = 1.5,
        verify: bool | str = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout_seconds),
            verify=verify,
            follow_redirects=True,
        )

    # --- lifecycle --------------------------------------------------------

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "FrigateClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # --- transport internals ----------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _url(self, path: str) -> str:
        if path.startswith("http"):
            return path
        return f"{self.base_url}{path}"

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        url = self._url(path)
        headers = self._headers()
        attempt = 0
        while True:
            try:
                response = self._client.request(method, url, headers=headers, params=params, timeout=timeout)
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
                if attempt < self.max_retries:
                    attempt += 1
                    debug(
                        "frigate request retrying",
                        method=method,
                        path=path,
                        attempt=attempt,
                        reason=str(exc),
                    )
                    self._backoff(attempt)
                    continue
                raise FrigateAPIError(
                    f"transport error contacting Frigate: {exc}",
                    url=url,
                    retryable=True,
                ) from exc

            if response.status_code in _RETRYABLE_STATUS and attempt < self.max_retries:
                attempt += 1
                debug(
                    "frigate request retrying",
                    method=method,
                    path=path,
                    attempt=attempt,
                    status=response.status_code,
                )
                self._backoff(attempt)
                continue

            if response.status_code >= 400:
                detail = self._error_detail(response)
                raise FrigateAPIError(
                    detail or f"Frigate API error",
                    status_code=response.status_code,
                    url=url,
                    retryable=response.status_code in _RETRYABLE_STATUS,
                )
            return response

    def _backoff(self, attempt: int) -> None:
        delay = self.retry_backoff * (2 ** (attempt - 1))
        time.sleep(delay)

    @staticmethod
    def _error_detail(response: httpx.Response) -> str:
        try:
            payload = response.json()
            if isinstance(payload, dict):
                msg = payload.get("message") or payload.get("detail")
                if msg:
                    return str(msg)
                return str(payload)[:200]
        except ValueError:
            pass
        text = response.text.strip()
        return text[:200] if text else ""

    def _get_json(
        self, path: str, params: dict[str, Any] | None = None
    ) -> Any:
        response = self._request("GET", path, params)
        try:
            return response.json()
        except ValueError as exc:
            raise FrigateAPIError(
                "Frigate returned non-JSON body",
                status_code=response.status_code,
                url=path,
                retryable=False,
            ) from exc

    # --- reviews ----------------------------------------------------------

    def list_reviews(
        self,
        after: float,
        before: float | None = None,
        cameras: Sequence[str] | None = None,
        labels: Sequence[str] | None = None,
        zones: Sequence[str] | None = None,
        severity: str | Sequence[str] | None = None,
        limit: int | None = None,
        page_size: int = 500,
    ) -> Iterable[Review]:
        """Iterate review items in a time window (start_time descending).

        The 0.18 endpoint returns a flat array with no paging object, so this
        windows on the ``before`` cursor (latest start_time seen). Items ending
        after ``after`` overlap the window boundary by design — cursor windows
        are handled inside this adapter only.
        """
        if after is None:
            raise ValueError("'after' is required")
        if severity:
            if isinstance(severity, str):
                severity_list = [severity]
            else:
                severity_list = list(severity)
        else:
            severity_list = []

        # Frigate filters per-severity; query them separately so the cursor
        # stays meaningful.
        for sev in severity_list or [None]:
            yield from self._paginate_reviews(
                after=after,
                before=before,
                cameras=cameras,
                labels=labels,
                zones=zones,
                severity=sev,
                limit=limit,
                page_size=page_size,
            )

    def _paginate_reviews(
        self,
        after: float,
        before: float | None,
        cameras: Sequence[str] | None,
        labels: Sequence[str] | None,
        zones: Sequence[str] | None,
        severity: str | None,
        limit: int | None,
        page_size: int,
    ) -> Iterable[Review]:
        page = min(page_size, limit) if limit else page_size
        cursor = before if before is not None else time.time()
        collected = 0
        while True:
            params: dict[str, Any] = {
                "after": after,
                "before": cursor,
                "limit": page,
            }
            if cameras:
                params["cameras"] = ",".join(cameras)
            if labels:
                params["labels"] = ",".join(labels)
            if zones:
                params["zones"] = ",".join(zones)
            if severity:
                params["severity"] = severity

            batch = self._get_json("/api/review", params)
            if not isinstance(batch, list):
                raise FrigateAPIError(
                    "unexpected /api/review payload (expected array)",
                    url="/api/review",
                )
            if not batch:
                break
            debug("frigate review page", count=len(batch), cursor=cursor)

            oldest = None
            for raw in batch:
                review = parse_review(raw)
                yield review
                collected += 1
                if limit is not None and collected >= limit:
                    return
                if oldest is None:
                    oldest = review.start_time
                oldest = min(oldest, review.start_time)

            if len(batch) < page:
                break
            if oldest is None or oldest <= after:
                break
            if cursor == oldest:
                # server ignored the cursor / equal timestamps: avoid infinite loop
                logger.warning("review cursor made no progress; stopping")
                break
            cursor = oldest

    def get_review(self, review_id: str) -> Review:
        payload = self._get_json(f"/api/review/{review_id}")
        if isinstance(payload, dict) and "id" in payload:
            return parse_review(payload)
        if isinstance(payload, list) and payload:
            return parse_review(payload[0])
        raise FrigateAPIError(
            "unexpected /api/review/{id} payload",
            url=f"/api/review/{review_id}",
        )

    # --- events -----------------------------------------------------------

    def list_events(
        self,
        after: float | None = None,
        before: float | None = None,
        cameras: Sequence[str] | None = None,
        labels: Sequence[str] | None = None,
        limit: int | None = None,
        page_size: int = 500,
    ) -> Iterable[Event]:
        """Iterate events (start_time descending), windowed on ``before``.

        Used mainly for standalone event inspection; the collector prefers the
        review-driven path (review -> event ids -> event detail).
        """
        page = min(page_size, limit) if limit else page_size
        cursor = before
        collected = 0
        while True:
            params: dict[str, Any] = {"limit": page, "sort": "date_desc"}
            if after is not None and cursor is None:
                params["after"] = after
            if cursor is not None:
                params["before"] = cursor
            if cameras:
                params["cameras"] = ",".join(cameras)
            if labels:
                params["labels"] = ",".join(labels)

            batch = self._get_json("/api/events", params)
            if not isinstance(batch, list):
                raise FrigateAPIError(
                    "unexpected /api/events payload (expected array)", url="/api/events"
                )
            if not batch:
                break
            debug("frigate events page", count=len(batch), cursor=cursor)

            oldest = None
            for raw in batch:
                event = parse_event(raw)
                yield event
                collected += 1
                if limit is not None and collected >= limit:
                    return
                if oldest is None:
                    oldest = event.start_time
                oldest = min(oldest, event.start_time)

            if len(batch) < page:
                break
            if after is not None and oldest is not None and oldest <= after:
                break
            if cursor == oldest:
                logger.warning("events cursor made no progress; stopping")
                break
            cursor = oldest

    def get_event(self, event_id: str) -> Event:
        payload = self._get_json(f"/api/events/{event_id}")
        if isinstance(payload, dict) and "id" in payload:
            return parse_event(payload)
        raise FrigateAPIError(
            "unexpected /api/events/{id} payload",
            url=f"/api/events/{event_id}",
        )

    # --- media downloads --------------------------------------------------

    def download_clean_snapshot(
        self, event_id: str, output_path: Any, timestamp: float | None = None
    ) -> str:
        """Download the clean (unannotated) snapshot for an event.

        Frigate 0.18 has no ``snapshot-clean.webp`` endpoint; the clean frame is
        ``/api/events/{id}/snapshot.jpg?bbox=0&timestamp=0`` (see snapshots.py).
        ``timestamp`` selects a specific frame in the event for temporal
        sampling; ``None``/0 keeps the default behavior. Returns the path written.
        """
        params = clean_snapshot_params()
        if timestamp is not None and timestamp > 0:
            # 0.18 parses timestamp as int; floats get 422.
            params["timestamp"] = int(timestamp)
        return self._download(
            clean_snapshot_url(event_id),
            output_path,
            params=params,
            timeout=self.timeout,
        )

    def download_event_snapshot(self, event_id: str, output_path: Any) -> str:
        """Download the annotated snapshot for an event (debug only)."""
        return self._download(
            annotated_snapshot_url(event_id),
            output_path,
            params=annotated_snapshot_params(),
            timeout=self.timeout,
        )

    def download_region_crop(
        self,
        event_id: str,
        output_path: Any,
        height: int,
        timestamp: float | None = None,
    ) -> str:
        """Download a server-side region crop (clean, no overlays) for an event.

        ``height`` is the requested output height in pixels; ``crop=1`` crops to
        the event bounding box. See snapshots.py.
        """
        params = region_crop_params(height)
        if timestamp is not None and timestamp > 0:
            # 0.18 parses timestamp as int; floats get 422.
            params["timestamp"] = int(timestamp)
        return self._download(
            clean_snapshot_url(event_id),
            output_path,
            params=params,
            timeout=self.timeout,
        )

    def download_annotated_crop(self, event_id: str, output_path: Any) -> str:
        """Download an annotated region crop (debug only)."""
        return self._download(
            annotated_snapshot_url(event_id),
            output_path,
            params=annotated_region_crop_params(),
            timeout=self.timeout,
        )

    def download_review_preview(
        self, review_id: str, output_path: Any, format: str = "mp4"
    ) -> str:
        """Download a review preview clip (gif or mp4)."""
        return self._download(
            review_preview_url(review_id, format),
            output_path,
            timeout=self.timeout,
        )

    def _download(
        self,
        path: str,
        output_path: Any,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> str:
        import pathlib

        destination = pathlib.Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        response = self._request("GET", path, params, timeout=timeout)
        content = response.content
        if not content:
            raise FrigateAPIError("empty download", url=path, retryable=True)
        destination.write_bytes(content)
        debug("frigate download", path=path, bytes=len(content), dest=destination.name)
        return str(destination.resolve())

    # --- motion -----------------------------------------------------------

    def get_motion_activity(
        self,
        after: float,
        before: float | None = None,
        cameras: Sequence[str] | None = None,
        scale: int = 30,
    ) -> list[MotionBucket]:
        params: dict[str, Any] = {"after": after, "scale": scale}
        if before is not None:
            params["before"] = before
        if cameras:
            params["cameras"] = ",".join(cameras)
        payload = self._get_json(motion_activity_url(), params)
        if not isinstance(payload, list):
            raise FrigateAPIError(
                "unexpected /api/review/activity/motion payload",
                url=motion_activity_url(),
            )
        return parse_motion_activity(payload)


__all__ = ["FrigateClient", "FrigateAPIError"]