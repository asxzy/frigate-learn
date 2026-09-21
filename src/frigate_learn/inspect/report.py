"""Static HTML dataset report (Phase 2).

``collect_records`` gathers sample metadata from SQLite; ``render_html_report``
writes a self-contained ``index.html`` plus small thumbnails, grouped by camera
then day, with the current quality verdict shown and machine-readable triage
instructions printed (triage itself is a separate CLI command that writes back
to ``samples.quality``).
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from ..config import AppConfig
from ..db import Database
from ..models import Annotation, Sample

THUMB_MAX_WIDTH = 480


@dataclass
class SampleRecord:
    sample_id: str
    camera: str
    timestamp: float
    label: str | None
    score: float | None
    image_path: str | None
    quality: str | None
    status: str
    verified: int


@dataclass
class ReportSummary:
    total: int = 0
    cameras: list[str] = None  # type: ignore[assignment]
    out_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.cameras is None:
            self.cameras = []


def collect_records(
    config: AppConfig,
    db: Database,
    *,
    cameras: list[str] | None = None,
    days: int | None = None,
    labels: list[str] | None = None,
    quality: str | None = None,
) -> list[SampleRecord]:
    """Query samples for the report, newest first."""
    from_ts: float | None = None
    if days is not None:
        from_ts = datetime.now(timezone.utc).timestamp() - days * 86400.0

    query = db.session().query(Sample)
    if cameras:
        query = query.filter(Sample.camera.in_(cameras))
    if labels:
        with_label = (
            db.session().query(Annotation.sample_id)
            .filter(Annotation.label.in_(labels))
        )
        query = query.filter(Sample.id.in_(with_label))
    if quality is not None:
        query = query.filter(Sample.quality == quality)
    if from_ts is not None:
        query = query.filter(Sample.timestamp >= from_ts)

    records = []
    for s in query.order_by(Sample.timestamp.desc()).limit(50000).all():
        records.append(
            SampleRecord(
                sample_id=s.id,
                camera=s.camera,
                timestamp=s.timestamp,
                label=s.frigate_label,
                score=s.frigate_score,
                image_path=s.image_path,
                quality=s.quality,
                status=s.status,
                verified=s.verified,
            )
        )
    return records


def _thumb(record: SampleRecord, out_dir: Path) -> str:
    """Return the relative thumb path for a record; generate it if needed."""
    thumb_dir = out_dir / "thumbs"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    rel = f"thumbs/{record.sample_id}.jpg"
    thumb = thumb_dir / f"{record.sample_id}.jpg"
    if not thumb.exists() and record.image_path and Path(record.image_path).is_file():
        try:
            with Image.open(record.image_path) as img:
                img = img.convert("RGB")
                scale = THUMB_MAX_WIDTH / max(1, img.width)
                if scale < 1.0:
                    img = img.resize(
                        (int(img.width * scale), int(img.height * scale)),
                        Image.Resampling.LANCZOS,
                    )
                img.save(thumb, quality=82)
        except Exception:
            return ""
    return rel if thumb.exists() else ""


def _grouped(records: list[SampleRecord]) -> list[tuple[str, list[tuple[str, list[SampleRecord]]]]]:
    """Group by camera, then local day of the timestamp."""
    by_camera: dict[str, dict[str, list[SampleRecord]]] = {}
    for rec in sorted(records, key=lambda r: r.timestamp):
        day = datetime.fromtimestamp(rec.timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
        by_camera.setdefault(rec.camera, {}).setdefault(day, []).append(rec)
    grouped = []
    for camera in sorted(by_camera):
        days = [(day, by_camera[camera][day]) for day in sorted(by_camera[camera], reverse=True)]
        grouped.append((camera, days))
    return grouped


def render_html_report(
    records: list[SampleRecord],
    out_dir: Path,
    *,
    title: str = "Frigate Learn — dataset inspector",
) -> ReportSummary:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    snippets = []
    for camera, days in _grouped(records):
        snippets.append(f"<section><h2>{html.escape(camera)} <small>({sum(len(d) for _, d in days)} frames)</small></h2>")
        for day, day_records in days:
            snippets.append(f"<h3>{html.escape(day)}</h3>")
            snippets.append('<div class="grid">')
            for rec in day_records:
                thumb = _thumb(rec, out_dir)
                tag = html.escape(rec.quality or "unset")
                time_s = html.escape(
                    datetime.fromtimestamp(rec.timestamp, tz=timezone.utc).strftime("%H:%M:%S")
                )
                img = f'<img src="{thumb}" loading="lazy" alt="{rec.sample_id[:8]}">' if thumb else ""
                score_s = f"{rec.score:.2f}" if rec.score is not None else "--"
                snippets.append(
                        "<figure>"
                        f"{img}"
                        f"<figcaption>"
                        f"{rec.sample_id[:8]} · {html.escape(rec.label or '?')} "
                        f"({score_s}) · {time_s}<br>"
                    f"<span class='q {tag}'>q={tag}</span> · status={rec.status}"
                    f"</figcaption>"
                    "</figure>"
                )
            snippets.append("</div>")
        snippets.append("</section>")

    css = (
        "body{font-family:system-ui;margin:2rem;color:#222}"
        "h2{border-bottom:2px solid #ddd;padding-bottom:.3rem}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}"
        "figure{margin:0;padding:.4rem;border:1px solid #eee;border-radius:6px}"
        "img{width:100%;height:130px;object-fit:cover;border-radius:4px}"
        "figcaption{font-size:12px;color:#555;margin-top:.3rem}"
        ".q{padding:1px 5px;border-radius:3px;background:#eee}"
        ".q.useful{background:#b7efcd}.q.bad{background:#f2b6b6}"
        ".q.duplicate{background:#ffe6a7}.q.ignore{background:#ccc}"
    )
    summary = ReportSummary(total=len(records), cameras=sorted(set(r.camera for r in records)))
    body = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title><style>{css}</style></head><body>"
        f"<h1>{html.escape(title)}</h1><p>{len(records)} frames · "
        f"cameras: {', '.join(html.escape(c) for c in summary.cameras)}</p>"
        + "".join(snippets)
        + "</body></html>"
    )
    (out_dir / "index.html").write_text(body, encoding="utf-8")
    summary.out_dir = out_dir
    return summary


__all__ = ["SampleRecord", "ReportSummary", "collect_records", "render_html_report"]