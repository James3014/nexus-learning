"""Pins for helpers de-duplicated in #78 (hash formats, shared _text, status sets)."""

from __future__ import annotations

import pytest

from nexus_learning import (
    adoption,
    campaign_closeout,
    campaign_gates,
    effectiveness_measurement,
    experiment_integrity,
    research_frontier,
)
from nexus_learning._statuses import PASS_STATUSES
from nexus_learning._textutil import require_non_empty_text

_FIXED = {"b": [1, "é"], "a": None}
_CLOSEOUT_HASH = "sha256:f8f17faab95c024891d173fa43442b0e52007736a1f36715ac721ab22deeefc5"
_INTEGRITY_HASH = "a44633ba036df25280e4fd9ef09da6f54dfe6d3611c755e31f993ab16f108f34"


def test_hash_formats_are_pinned_and_intentionally_different():
    assert campaign_closeout._hash(_FIXED) == _CLOSEOUT_HASH
    assert experiment_integrity._hash(_FIXED) == _INTEGRITY_HASH
    assert campaign_closeout._hash({"x": 1}).startswith("sha256:")
    assert not experiment_integrity._hash({"x": 1}).startswith("sha256:")


@pytest.mark.parametrize("bad", [None, 5, b"x", "", "   ", ["a"]])
def test_shared_text_rejects_non_strings_and_blank(bad):
    with pytest.raises(ValueError, match="f must be a non-empty string"):
        require_non_empty_text(bad, "f")


def test_shared_text_strips():
    assert require_non_empty_text("  a b \n", "f") == "a b"


def test_modules_use_shared_text():
    assert campaign_gates._text is require_non_empty_text
    assert research_frontier._text is require_non_empty_text


def test_pass_status_set_is_shared_object():
    assert adoption._PASS_STATUSES is PASS_STATUSES
    assert effectiveness_measurement._PASS_STATUSES is PASS_STATUSES
    assert PASS_STATUSES == {"pass", "passed", "success", "succeeded", "qualified"}
