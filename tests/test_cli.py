"""Tests for the CLI surface."""

import io
from contextlib import redirect_stdout

from eureqai.cli import main


def _run(argv):
    return main(argv)


def test_init_writes_a_template(tmp_path):
    target = tmp_path / "eureqai.yml"
    assert _run(["init", "--output", str(target), "--role", "provider"]) == 0
    text = target.read_text()
    assert "project:" in text
    assert "answers:" in text
    assert "PROHIB-1" in text


def test_init_respects_force(tmp_path):
    target = tmp_path / "eureqai.yml"
    target.write_text("existing")
    rc = _run(["init", "--output", str(target)])
    assert rc == 2
    assert target.read_text() == "existing"
    assert _run(["init", "--output", str(target), "--force"]) == 0
    assert "project:" in target.read_text()


def test_assess_emits_markdown_to_stdout(tmp_path):
    config = tmp_path / "eureqai.yml"
    _run(["init", "--output", str(config), "--role", "provider"])
    # Default template answers everything "no" — perfect for testing.
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = _run(["assess", "--config", str(config)])
    assert rc == 0
    assert "AI Act readiness report" in buf.getvalue()


def test_assess_fail_on_blockers_exits_nonzero(tmp_path):
    config = tmp_path / "eureqai.yml"
    _run(["init", "--output", str(config), "--role", "provider"])
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = _run(["assess", "--config", str(config), "--fail-on-blockers"])
    assert rc == 1


def test_catalogue_lists_items(capsys):
    rc = _run(["catalogue"])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "PROHIB-1" in captured
    assert "TECH-1" in captured
