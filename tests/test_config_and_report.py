"""Tests for the YAML config schema, assessment, and report renderer."""

import textwrap

import pytest

from eureqai.config import ProjectConfig, load_config
from eureqai.report import assess, render_markdown


def _write_yaml(tmp_path, body):
    path = tmp_path / "project.yml"
    path.write_text(textwrap.dedent(body))
    return path


VALID_YAML = """\
project:
  name: TestSystem
  version: 1.0.0
  intended_purpose: Demo
classification:
  role: provider
  risk_class: high_risk
answers:
  PROHIB-1: yes
  RISK-1: yes
  DOC-1: partial
  TECH-1: no
  OVER-1: yes
  TRANS-1: yes
documents:
  technical_documentation: docs/annex_iv.md
"""


def test_load_config_accepts_valid_yaml(tmp_path):
    config = load_config(_write_yaml(tmp_path, VALID_YAML))
    assert isinstance(config, ProjectConfig)
    assert config.project.name == "TestSystem"
    assert config.classification.role == "provider"
    assert config.answers["PROHIB-1"] == "yes"


def test_load_config_rejects_invalid_answer(tmp_path):
    bad = VALID_YAML.replace("PROHIB-1: yes", "PROHIB-1: maybe")
    with pytest.raises(Exception):
        load_config(_write_yaml(tmp_path, bad))


def test_load_config_rejects_invalid_role(tmp_path):
    bad = VALID_YAML.replace("role: provider", "role: spectator")
    with pytest.raises(Exception):
        load_config(_write_yaml(tmp_path, bad))


def test_assess_groups_by_area_and_flags_unanswered(tmp_path):
    config = load_config(_write_yaml(tmp_path, VALID_YAML))
    assessment = assess(config)
    areas = {a.area: a for a in assessment.areas}
    assert "Prohibited practices" in areas
    assert "Technical robustness" in areas
    # We left several provider items unanswered.
    assert assessment.unanswered


def test_assess_critical_no_marks_overall_not_ready(tmp_path):
    config = load_config(_write_yaml(tmp_path, VALID_YAML))
    assessment = assess(config)
    # TECH-1 is critical and answered "no" → overall must be not_ready.
    assert assessment.overall_level == "not_ready"


def test_render_markdown_contains_expected_sections(tmp_path):
    config = load_config(_write_yaml(tmp_path, VALID_YAML))
    md = render_markdown(assess(config))
    assert "AI Act readiness report" in md
    assert "## Summary" in md
    assert "## Detailed findings" in md
    assert "not legal advice" in md
