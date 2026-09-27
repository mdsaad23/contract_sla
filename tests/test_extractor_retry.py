"""The retry guard in extract_sla must rescue a flaky response without
rescuing a model that is genuinely bad at JSON.

  python tests/test_extractor_retry.py
"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pipeline.extractor as X
from pipeline.llm import LLMResponse

TRUNCATED = LLMResponse('{"governing_law": "New Y', 10, 5, 0.1, 0.0, 0)
GOOD = LLMResponse('{"governing_law": "New York", "liability_cap": "USD 1m"}', 10, 9, 0.1, 0.0, 0)


def test_flaky_response_is_retried():
    with patch.object(X, "call_llm", side_effect=[TRUNCATED, GOOD]):
        r = X.extract_sla("c", "nope.txt", "ctx")
    assert r.status == "success"
    assert r.sla.governing_law == "New York"
    # the metrics must come from the attempt that was actually used
    assert r.completion_tokens == 9


def test_discarded_attempt_is_still_billed():
    with patch.object(X, "call_llm", side_effect=[TRUNCATED._replace(cost_usd=0.01),
                                                  GOOD._replace(cost_usd=0.02)]):
        r = X.extract_sla("c", "nope.txt", "ctx")
    assert abs(r.cost_usd - 0.03) < 1e-9


def test_failed_extraction_keeps_its_cost():
    """Both attempts malformed: nothing usable came back, but both were billed."""
    with patch.object(X, "call_llm", side_effect=[TRUNCATED._replace(cost_usd=0.01)] * 2):
        r = X.extract_sla("c", "nope.txt", "ctx")
    assert r.status == "failed"
    assert abs(r.cost_usd - 0.02) < 1e-9


def test_billed_error_keeps_its_cost():
    """A reasoning-only response raises, after the first attempt was already paid."""
    err = RuntimeError("reasoning-only response")
    err.cost_usd = 0.05
    with patch.object(X, "call_llm", side_effect=[TRUNCATED._replace(cost_usd=0.01), err]):
        r = X.extract_sla("c", "nope.txt", "ctx")
    assert r.status == "failed"
    assert abs(r.cost_usd - 0.06) < 1e-9


def test_schema_failure_after_a_good_call_keeps_its_cost():
    bad = LLMResponse('{"governing_law": {}}', 10, 9, 0.1, 0.0, 0, cost_usd=0.04)
    with patch.object(X, "call_llm", return_value=bad),          patch.object(X, "SLAClause", side_effect=[ValueError("bad"), X.SLAClause()]):
        r = X.extract_sla("c", "nope.txt", "ctx")
    assert r.status == "failed"
    assert abs(r.cost_usd - 0.04) < 1e-9


def test_consistently_malformed_still_fails():
    """A temperature=0 local model returns the same bytes twice, so bad JSON
    stays a failure and the benchmark keeps measuring it as one."""
    with patch.object(X, "call_llm", side_effect=[TRUNCATED, TRUNCATED]):
        r = X.extract_sla("c", "nope.txt", "ctx")
    assert r.status == "failed"
    assert "JSONDecodeError" in r.error


if __name__ == "__main__":
    test_flaky_response_is_retried()
    test_discarded_attempt_is_still_billed()
    test_failed_extraction_keeps_its_cost()
    test_billed_error_keeps_its_cost()
    test_schema_failure_after_a_good_call_keeps_its_cost()
    test_consistently_malformed_still_fails()
    print("ok")
