"""The CLI scripts that carry a little logic of their own. Nothing here sends a request."""

import importlib.util

import pytest

from fukidashi import config
from fukidashi.llm.client import LLMConfig

spec = importlib.util.spec_from_file_location("m0_first_call", config.ROOT / "scripts" / "m0_first_call.py")
m0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m0)

PRICED = LLMConfig("http://provider.test/v1", "key", "mock-model", price_input_per_m=1.0, price_output_per_m=2.0)
UNPRICED = LLMConfig("http://provider.test/v1", "key", "mock-model")
HINT = "set LLM_PRICE_INPUT_PER_M / LLM_PRICE_OUTPUT_PER_M in .env"


def summary(*, calls=1, unknown=0, cost=None):
    return {"calls": calls, "unknown_usage_calls": unknown, "input_tokens": 0, "output_tokens": 0, "cost_usd": cost}


def test_a_known_cost_is_shown_as_dollars():
    assert m0.cost_line(summary(cost=0.00014), PRICED) == "Cost: $0.0001"


@pytest.mark.parametrize("unknown, wording", [(1, "1 call returned no usage"), (3, "3 calls returned no usage")])
def test_an_unknown_cost_with_prices_set_says_unknown_and_how_many_calls_had_no_usage(unknown, wording):
    line = m0.cost_line(summary(calls=3, unknown=unknown), PRICED)

    assert line == f"Cost: unknown ({wording})"
    assert HINT not in line  # the prices are fine: pointing at .env would mislead


def test_a_missing_price_still_shows_the_price_hint():
    assert m0.cost_line(summary(), UNPRICED) == f"Cost: not computed ({HINT})"
    half = LLMConfig("http://provider.test/v1", "key", "mock-model", price_input_per_m=1.0)
    assert HINT in m0.cost_line(summary(), half)


def test_both_reasons_are_shown_when_both_apply():
    line = m0.cost_line(summary(calls=2, unknown=1), UNPRICED)

    assert "unknown (1 call returned no usage)" in line and HINT in line
