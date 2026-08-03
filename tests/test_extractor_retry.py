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


def test_consistently_malformed_still_fails():
    """A temperature=0 local model returns the same bytes twice, so bad JSON
    stays a failure and the benchmark keeps measuring it as one."""
    with patch.object(X, "call_llm", side_effect=[TRUNCATED, TRUNCATED]):
        r = X.extract_sla("c", "nope.txt", "ctx")
    assert r.status == "failed"
    assert "JSONDecodeError" in r.error


if __name__ == "__main__":
    test_flaky_response_is_retried()
    test_consistently_malformed_still_fails()
    print("ok")
