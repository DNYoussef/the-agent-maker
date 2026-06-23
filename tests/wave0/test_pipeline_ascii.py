"""Orchestrator pipeline.py must be pure ASCII: it prints during real runs and
the unicode glyphs crashed on Windows cp1252 stdout (the PYTHONIOENCODING=utf-8
workaround should not be load-bearing). Global rule: no unicode in source."""

from pathlib import Path


def test_pipeline_source_is_ascii():
    path = (
        Path(__file__).resolve().parents[2] / "src" / "cross_phase" / "orchestrator" / "pipeline.py"
    )
    text = path.read_text(encoding="utf-8")
    offenders = [
        i + 1 for i, line in enumerate(text.splitlines()) if any(ord(c) > 127 for c in line)
    ]
    assert not offenders, f"non-ASCII characters in pipeline.py at lines {offenders}"
