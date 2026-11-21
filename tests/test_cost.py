from tool_agent.cost import RunBudget, estimate_tokens, run_cost, token_cost


def test_estimate_tokens_uses_four_chars():
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcdefgh") == 2
    assert estimate_tokens("") == 0


def test_estimate_tokens_rejects_bad_ratio():
    try:
        estimate_tokens("x", chars_per_token=0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero ratio accepted")


def test_token_cost_math():
    assert token_cost(1_000_000, 2.5) == 2.5
    assert token_cost(500_000, 2.0) == 1.0


def test_token_cost_rejects_negative():
    for args in ((-1, 1.0), (1, -1.0)):
        try:
            token_cost(*args)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid input: {args}")


def test_run_budget_properties():
    budget = RunBudget(budget_tokens=100, used_tokens=120)
    assert budget.remaining == 0
    assert budget.exceeded is True
    assert budget.utilization == 1.2
    ok = RunBudget(budget_tokens=100, used_tokens=40)
    assert ok.remaining == 60
    assert ok.exceeded is False
    assert ok.utilization == 0.4


def test_run_budget_zero_budget():
    assert RunBudget(budget_tokens=0, used_tokens=0).utilization == 0.0
    assert RunBudget(budget_tokens=0, used_tokens=5).utilization == 1.0


def test_run_cost_report():
    report = run_cost(1000, 500, input_price_per_1m=1.0, output_price_per_1m=2.0)
    assert report["total_tokens"] == 1500
    assert report["input_cost_usd"] == 0.001
    assert report["output_cost_usd"] == 0.001
    assert report["total_cost_usd"] == 0.002


def test_run_cost_rejects_negative_tokens():
    try:
        run_cost(-1, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("negative tokens accepted")
