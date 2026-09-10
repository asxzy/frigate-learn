"""CLI tests (offline; Collector is stubbed)."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from frigate_learn.cli import cli
from frigate_learn.collection.collector import CollectSummary
from conftest import make_config_file

CONFIG_BODY = """\
frigate:
  base_url: "http://frigate:8971"
  token: "${TEST_FRIGATE_TOKEN:-none}"
data:
  root: "data"
"""


@pytest.fixture
def dbenv(tmp_path):
    cfg = tmp_path / "config.yaml"
    make_config_file(cfg, CONFIG_BODY)
    return {"FRIGATE_LEARN_CONFIG": str(cfg)}


def test_version(dbenv):
    result = CliRunner().invoke(cli, ["--version"], env=dbenv)
    assert result.exit_code == 0
    assert "0.2.0" in result.output


def test_db_migrate_and_status(dbenv):
    runner = CliRunner()
    migrate = runner.invoke(cli, ["db", "migrate"], env=dbenv)
    assert migrate.exit_code == 0
    assert "0001_initial.sql" in migrate.output
    assert "0002_multiframe_phash.sql" in migrate.output

    status = runner.invoke(cli, ["status"], env=dbenv)
    assert status.exit_code == 0
    assert "Database:" in status.output
    assert "Schema:" in status.output
    assert "Samples:" in status.output


def test_db_stats_after_second_migrate_is_noop(dbenv):
    runner = CliRunner()
    runner.invoke(cli, ["db", "migrate"], env=dbenv)
    again = runner.invoke(cli, ["db", "migrate"], env=dbenv)
    assert "No pending migrations." in again.output


def test_collect_output(dbenv, monkeypatch):
    class FakeCollector:
        def __init__(self, config, database):
            self.client = type("C", (), {"close": lambda self: None})()

        def collect(self, from_ts, to_ts=None, cameras=None, labels=None, severity=None,
                    limit=None, concurrency=None, progress=None):
            if progress:
                progress("Reviews found: 2")
            return CollectSummary(
                reviews_found=2,
                reviews_selected=2,
                events_found=2,
                events_new=2,
                new_samples=2,
                duplicate_samples=0,
                failures=0,
                new_annotations=2,
                range_from=from_ts,
                range_to=to_ts,
                duration_seconds=0.1,
            )

    monkeypatch.setattr("frigate_learn.cli.Collector", FakeCollector)
    result = CliRunner().invoke(cli, ["collect", "--from", "1 day ago"], env=dbenv)
    assert result.exit_code == 0
    assert "Reviews found: 2" in result.output
    assert "New samples: 2" in result.output
    assert "Done in 0.1s." in result.output


def test_collect_no_region_crop_flag(dbenv, monkeypatch):
    captured = {}

    class RecordingCollector:
        def __init__(self, config, database):
            captured["region_crop"] = config.collection.region_crop
            self.client = type("C", (), {"close": lambda self: None})()

        def collect(self, from_ts, to_ts=None, cameras=None, labels=None, severity=None,
                    limit=None, concurrency=None, progress=None):
            return CollectSummary(new_samples=0)

    monkeypatch.setattr("frigate_learn.cli.Collector", RecordingCollector)
    result = CliRunner().invoke(cli, ["collect", "--no-region-crop"], env=dbenv)
    assert result.exit_code == 0
    assert captured["region_crop"] is False


def test_collect_bad_severity(dbenv, monkeypatch):
    monkeypatch.setattr("frigate_learn.cli.Collector", lambda config, db: None)
    result = CliRunner().invoke(cli, ["collect", "--severity", "bogus"], env=dbenv)
    assert result.exit_code != 0
    assert "bogus" in result.output.lower() or "invalid" in result.output.lower()


def test_run_pipeline_smoke(dbenv):
    result = CliRunner().invoke(
        cli, ["run", "--steps", "build", "--steps", "benchmark", "--until", "gate"],
        env=dbenv,
    )
    assert result.exit_code == 0
    assert "[ok ] build" in result.output or "[-- ] benchmark" in result.output


def test_run_rejects_unknown_stage(dbenv):
    result = CliRunner().invoke(cli, ["run", "--steps", "nope"], env=dbenv)
    assert result.exit_code != 0
    assert "unknown stage" in result.output.lower()