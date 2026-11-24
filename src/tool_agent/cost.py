"""Run-level tokenomics (agentic-ai ch.17 cross-pollination).

The agent loop already bounds output characters; this module converts that
into token and cost estimates so a run can be compared on unit economics,
not just step count.
"""

from __future__ import annotations

from dataclasses import dataclass


def estimate_tokens(text: str, *, chars_per_token: float = 4.0) -> int:
    """Cheap, dependency-free token estimate (4 chars/token is a safe default)."""
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be positive")
    if not text:
        return 0
    return max(1, round(len(text) / chars_per_token))


def token_cost(tokens: int, price_per_1m: float) -> float:
    """USD cost for ``tokens`` at ``price_per_1m`` dollars per million tokens."""
    if tokens < 0:
        raise ValueError("tokens must be >= 0")
    if price_per_1m < 0:
        raise ValueError("price must be >= 0")
    return round(tokens * price_per_1m / 1_000_000, 6)


@dataclass(frozen=True, slots=True)
class RunBudget:
    budget_tokens: int
    used_tokens: int

    def __post_init__(self) -> None:
        if self.budget_tokens < 0 or self.used_tokens < 0:
            raise ValueError("token counts must be non-negative")

    @property
    def remaining(self) -> int:
        return max(self.budget_tokens - self.used_tokens, 0)

    @property
    def exceeded(self) -> bool:
        return self.used_tokens > self.budget_tokens

    @property
    def utilization(self) -> float:
        if self.budget_tokens == 0:
            return 1.0 if self.used_tokens else 0.0
        return round(self.used_tokens / self.budget_tokens, 4)


def run_cost(
    input_tokens: int,
    output_tokens: int,
    *,
    input_price_per_1m: float = 0.0,
    output_price_per_1m: float = 0.0,
) -> dict:
    """Combine token counts and prices into a small cost report."""
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("token counts must be non-negative")
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_cost_usd": token_cost(input_tokens, input_price_per_1m),
        "output_cost_usd": token_cost(output_tokens, output_price_per_1m),
        "total_cost_usd": round(
            token_cost(input_tokens, input_price_per_1m)
            + token_cost(output_tokens, output_price_per_1m),
            6,
        ),
    }
