"""Frigate HTTP client tests (transport mocked with respx)."""

from __future__ import annotations

from pathlib import Path

import pytest
import respx
from httpx import Response

from frigate_learn.frigate.client import FrigateAPIError, FrigateClient


@pytest.fixture
def mock_api():
    with respx.mock(assert_all_called=False) as m:
        yield m


def make_client(**kw) -> FrigateClient:
    params = dict(base_url="http://frigate:8971", token="tok", max_retries=2, retry_backoff=0.01)
    params.update(kw)
    return FrigateClient(**params)


def _review_raw(r_id: str, start: float, detections=None) -> dict:
    return {
        "id": r_id,
        "camera": "front",
        "severity": "detection",
        "start_time": start,
        "end_time": start + 60,
        "thumb_path": "/x",
        "data": {"detections": detections or [r_id + "-evt"]},
    }


def test_auth_header(mock_api):
    mock_api.get("/api/review").mock(return_value=Response(200, json=[]))
    client = make_client()
    list(client.list_reviews(after=1))
    request = mock_api.calls[0].request
    assert request.headers["Authorization"] == "Bearer tok"
    client.close()


def test_list_reviews_windowed_two_pages(mock_api):
    route = mock_api.get("/api/review")
    page1 = [_review_raw("r1", 9000), _review_raw("r2", 8000)]
    page2 = [_review_raw("r3", 7000)]
    route.side_effect = [
        Response(200, json=page1),
        Response(200, json=page2),
    ]
    client = make_client()
    reviews = list(client.list_reviews(after=1000, page_size=2))
    assert [r.id for r in reviews] == ["r1", "r2", "r3"]
    assert len(mock_api.calls) == 2
    # cursor advanced to the oldest start_time seen
    assert float(mock_api.calls[1].request.url.params["before"]) == 8000
    assert float(mock_api.calls[1].request.url.params["after"]) == 1000
    client.close()


def test_list_reviews_limit(mock_api):
    mock_api.get("/api/review").mock(
        side_effect=[
            Response(200, json=[_review_raw("r1", 9000), _review_raw("r2", 8000)]),
        ]
    )
    client = make_client()
    reviews = list(client.list_reviews(after=1000, limit=1))
    assert [r.id for r in reviews] == ["r1"]
    client.close()


def test_list_reviews_per_severity(mock_api):
    mock_api.get("/api/review").mock(
        side_effect=[Response(200, json=[]), Response(200, json=[])]
    )
    client = make_client()
    list(client.list_reviews(after=1000, severity=["alert", "detection"]))
    assert len(mock_api.calls) == 2
    severities = {c.request.url.params["severity"] for c in mock_api.calls}
    assert severities == {"alert", "detection"}
    client.close()


def test_404_raises_not_retryable(mock_api):
    mock_api.get("/api/review").mock(return_value=Response(404, text="nope"))
    client = make_client()
    with pytest.raises(FrigateAPIError) as excinfo:
        list(client.list_reviews(after=1000))
    assert excinfo.value.status_code == 404
    assert excinfo.value.retryable is False
    assert len(mock_api.calls) == 1
    client.close()


def test_500_retries_then_raises(mock_api):
    mock_api.get("/api/review").mock(return_value=Response(500, text="oops"))
    client = make_client()
    with pytest.raises(FrigateAPIError) as excinfo:
        list(client.list_reviews(after=1000))
    assert excinfo.value.status_code == 500
    assert excinfo.value.retryable is True
    assert len(mock_api.calls) == 3  # initial + 2 retries (max_retries=2)
    client.close()


def test_get_event(mock_api):
    mock_api.get("/api/events/e1").mock(
        return_value=Response(
            200,
            json={
                "id": "e1",
                "camera": "front",
                "label": "person",
                "start_time": 100.0,
                "end_time": 160.0,
                "box": [0.1, 0.2, 0.4, 0.8],
                "data": {"score": 0.95},
            },
        )
    )
    client = make_client()
    event = client.get_event("e1")
    assert event.label == "person"
    assert event.box == (0.1, 0.2, 0.4, 0.8)
    assert event.completed is True
    client.close()


def test_download_clean_snapshot(mock_api, tmp_path):
    payload = b"\xff\xd8\xff\xe0fakejpeg"
    mock_api.get("/api/events/e1/snapshot.jpg").mock(
        return_value=Response(200, content=payload)
    )
    client = make_client()
    dst = client.download_clean_snapshot("e1", tmp_path / "imgs" / "e1.jpg")
    assert Path(dst).read_bytes() == payload
    params = mock_api.calls[0].request.url.params
    assert params["bbox"] == "0"
    assert params["timestamp"] == "0"
    client.close()


def test_empty_download_raises(mock_api, tmp_path):
    mock_api.get("/api/events/e1/snapshot.jpg").mock(return_value=Response(200, content=b""))
    client = make_client()
    with pytest.raises(FrigateAPIError):
        client.download_clean_snapshot("e1", tmp_path / "e1.jpg")
    client.close()


def test_motion_activity(mock_api):
    mock_api.get("/api/review/activity/motion").mock(
        return_value=Response(
            200,
            json=[
                {"start_time": 1000, "motion": 0.5, "camera": "front"},
                {"start_time": 1060, "motion": 0.0, "camera": "front"},
            ],
        )
    )
    client = make_client()
    buckets = client.get_motion_activity(after=999)
    assert [(b.start_time, b.motion) for b in buckets] == [(1000, 0.5), (1060, 0.0)]
    assert len(mock_api.calls) == 1
    params = mock_api.calls[0].request.url.params
    assert params["after"] == "999"
    assert params["scale"] == "30"
    client.close()


def test_non_json_body_raises(mock_api):
    mock_api.get("/api/review").mock(return_value=Response(200, text="<html>wat</html>"))
    client = make_client()
    with pytest.raises(FrigateAPIError):
        list(client.list_reviews(after=1000))
    client.close()


def test_list_events_windowed(mock_api):
    route = mock_api.get("/api/events")
    route.side_effect = [
        Response(
            200,
            json=[
                {"id": "e1", "camera": "front", "label": "person", "start_time": 9000},
                {"id": "e2", "camera": "front", "label": "car", "start_time": 8000},
            ],
        ),
        Response(
            200,
            json=[{"id": "e3", "camera": "front", "label": "person", "start_time": 7000}],
        ),
    ]
    client = make_client()
    events = list(client.list_events(after=6500, page_size=2))
    assert [e.id for e in events] == ["e1", "e2", "e3"]
    assert float(mock_api.calls[1].request.url.params["before"]) == 8000
    client.close()


def test_download_region_crop_params(mock_api, tmp_path):
    payload = b"\xff\xd8\xff\xe0fakecropjpeg"
    mock_api.get("/api/events/e1/snapshot.jpg").mock(
        return_value=Response(200, content=payload)
    )
    client = make_client()
    dst = client.download_region_crop("e1", tmp_path / "crop" / "e1.jpg", height=320)
    assert Path(dst).read_bytes() == payload
    params = mock_api.calls[0].request.url.params
    assert params["crop"] == "1"
    assert params["bbox"] == "0"
    assert params["timestamp"] == "0"
    assert params["height"] == "320"
    client.close()


def test_download_region_crop_timestamp_param(mock_api, tmp_path):
    mock_api.get("/api/events/e1/snapshot.jpg").mock(
        return_value=Response(200, content=b"x")
    )
    client = make_client()
    client.download_region_crop("e1", tmp_path / "e1.jpg", height=640, timestamp=123.0)
    assert mock_api.calls[0].request.url.params["timestamp"] == "123.0"
    client.close()


def test_download_annotated_crop_params(mock_api, tmp_path):
    mock_api.get("/api/events/e1/snapshot.jpg").mock(
        return_value=Response(200, content=b"x")
    )
    client = make_client()
    client.download_annotated_crop("e1", tmp_path / "e1-debug.jpg")
    params = mock_api.calls[0].request.url.params
    assert params["crop"] == "1"
    assert "bbox" not in params
    client.close()