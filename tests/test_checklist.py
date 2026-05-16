"""Tests for the readiness checklist catalogue."""

import pytest

from eureqai.checklist import (
    ANSWER_SCORE,
    CATALOGUE,
    PRIORITY_WEIGHT,
    items_for_role,
    score_area,
)


def test_catalogue_ids_are_unique():
    ids = [item.id for item in CATALOGUE]
    assert len(ids) == len(set(ids))


def test_every_item_has_an_article_and_priority():
    for item in CATALOGUE:
        assert item.article
        assert item.priority in PRIORITY_WEIGHT


def test_role_filter_provider_returns_full_catalogue():
    assert len(items_for_role("provider")) == len(CATALOGUE)


def test_role_filter_deployer_is_a_strict_subset():
    deployer = items_for_role("deployer")
    assert 0 < len(deployer) < len(CATALOGUE)


def test_role_filter_importer_is_minimal():
    importer = items_for_role("importer")
    assert {item.area for item in importer} <= {
        "Prohibited practices",
        "Technical documentation",
        "Transparency",
    }


def test_score_area_all_yes_is_one():
    items = [(CATALOGUE[0], "yes", "")]
    assert score_area(items) == pytest.approx(1.0)


def test_score_area_all_no_is_zero():
    items = [(CATALOGUE[0], "no", "")]
    assert score_area(items) == pytest.approx(0.0)


def test_score_area_partial_is_half():
    items = [(CATALOGUE[0], "partial", "")]
    assert score_area(items) == pytest.approx(0.5)


def test_na_counts_as_addressed():
    assert ANSWER_SCORE["na"] == 1.0
