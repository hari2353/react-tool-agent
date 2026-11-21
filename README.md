# ReAct Tool-Calling Agent (LLM-Protocol Design)

**The modern replacement for my old LangChain demo** — same idea, rebuilt the way production agent systems should be: a small ReAct loop over a **swappable LLM protocol**, with safe validated tools, retrieval, full step tracing, and a deterministic offline mode that needs zero credentials.

## The demo run (verified, runs in CI with no API keys)

```
$ python scripts/run_agent.py

QUESTION: How often are high-risk suppliers audited, and how far apart are
low-risk self-assessments in weeks? Also convert 100 miles to km ...

  step 1: tool: kb_search({"query": "audit cadence by risk band", "top_k": 2})
  step 1: -> [{"doc_id": "risk-bands", ...}, {"doc_id": "audit-cadence", ...}]
  step 2: tool: unit_convert({"value": 100, "unit": "mi", "target": "km"})
  step 2: -> 160.9344
  step 3: tool: calculator({"expression": "36 * 12"})
  step 3: -> 432.0
  step 4: final answer

SUCCESS   : True
TOOLS USED: ('kb_search', 'unit_convert', 'calculator')
```

Multi-step tool chaining with retrieval-grounded answers — deterministic on every machine.

**Verified quality bar:** 67 tests (unit + multi-step integration), **96% coverage**, ruff + mypy clean, CI on Python 3.10-3.12 including the demo run above.

## Why this design (the interview talking points)

| Choice | Why it matters |
|---|---|
| **`ChatLLM` protocol** | The agent loop depends on an interface, not a vendor. `ScriptedLLM` (deterministic replay) and `AzureOpenAILLM` (env-credential adapter) are one-line swaps at the composition root. No vendor lock, no deprecated framework imports. |
| **Deterministic offline mode** | Agent behavior is *testable*. The whole loop — tool dispatch, repair, retries, tracing — runs in CI with zero network, zero cost, zero flakiness. Most agent demos can't say this. |
| **Validated tool dispatch** | The registry checks every call against its `ToolSpec`: unknown tools, missing and unexpected arguments are rejected *before* execution. The LLM is never trusted blindly. |
| **Sandboxed calculator** | Arithmetic is evaluated on a whitelisted AST (no names, attributes, subscripts, comprehensions). `__import__` attempts raise, not execute. |
| **Self-repair loop** | Tool errors are fed back into the conversation so the LLM can fix its next call; three consecutive failures abort with a clean failure result. |
| **Bounded loop + immutable trace** | `max_steps` guarantees termination; every step lands in an immutable `AgentStep` trace (the observability story). |
| **Zero-dependency core** | stdlib only. `openai` is an optional extra for the Azure adapter. |

## The JSON contract

Both LLM implementations speak the same decision format:

```json
{"action": "tool",  "tool": "kb_search", "arguments": {"query": "...", "top_k": 2}}
{"action": "final", "answer": "..."}
```

Malformed output raises `LLMError` and is captured as a failed run with the reason in the trace — never an unhandled crash.

## Tools included

- **`kb_search`** — TF-IDF retrieval (sublinear TF, L2-normalized, cosine ranking) over a synthetic supplier-compliance corpus (same domain as my ESG pipeline repo)
- **`calculator`** — AST-whitelisted arithmetic (`+ - * / % // **`, `sqrt, abs, round, min, max, log, log10, exp`)
- **`unit_convert`** — km/mi/m, kg/lb, C/F

Adding a tool = one `ToolSpec` + one function + `registry.register(...)` — the system prompt regenerates itself, and the agent code never changes (open/closed).

## Quickstart

```bash
pip install -e .[dev]
pytest                       # 66 tests
python scripts/run_agent.py  # the demo above, offline
```

## Production loop hardening (market-steal, 2026-11-18)

- **Multi-tool decisions** — a single model turn may fan out to several
  calls: `{"action": "tool", "tools": [{...}, {...}]}`. Calls execute in
  order (each validated and traced as its own step pair); one LLM
  round-trip buys up to `max_tools_per_decision` (default 4) dispatches.
- **Run budget** — `AgentConfig.max_output_chars` (default 40k) caps the
  cumulative model output across a run: a deterministic cost proxy on top
  of `max_steps`. Exceeding it ends the run as a clean captured failure.
- **Bounded repair re-prompt** — a malformed model output is retried with
  a format reminder (up to `AgentConfig.max_retries`, default 1) instead
  of aborting the run on the first bad parse; exhausting retries still
  ends in a clean captured failure with the parse error.

### Going live with Azure OpenAI

```bash
pip install -e .[azure]
export AZURE_OPENAI_ENDPOINT="https://<res>.openai.azure.com"
export AZURE_OPENAI_API_KEY="<key>"
export AZURE_OPENAI_DEPLOYMENT="gpt-4o"
python scripts/run_agent.py --live
```

## Durable runs + approvals

```python
agent.run("q", checkpoint_path="run.json", approve=lambda call: call.tool != "delete")
agent.resume("run.json", approve=...)      # continues; applied calls never repeat
```

`JsonCheckpointStore` writes a versioned JSON snapshot atomically after every
completed step and clears it on a terminal result. A denied call stops with
`error="tool call denied by human approval"` and an `approval_denied` step
without executing. `AgentConfig(max_tool_retries=1)` retries `ToolError`s and
records each attempt. The checkpoint contains the conversation text, so treat
the file like the prompt itself.

## Run tokenomics (Harvard Edge agentic-ai cross-pollination)

`tool_agent.cost` converts the loop's character budgets into token and cost
estimates, so a run can be compared on unit economics, not just step count:

```python
from tool_agent.cost import RunBudget, estimate_tokens, run_cost

run_cost(estimate_tokens(prompt), estimate_tokens(answer), input_price_per_1m=1.0, output_price_per_1m=2.0)
RunBudget(budget_tokens=4000, used_tokens=used).remaining
```

No tokenizer dependency; 4 chars/token by default. Cross-pollinated from
agentic-ai ch.17 (tokenomics).

## Repository layout

```
src/tool_agent/
  models.py      # typed frozen domain: messages, tool specs/calls, steps, results
  llm.py         # ChatLLM protocol + JSON contract + ScriptedLLM + Azure adapter
  tools.py       # safe tools + validating ToolRegistry (+ system-prompt generation)
  retrieval.py   # TF-IDF store + synthetic compliance corpus
  agent.py       # the ReAct loop: bounded steps, repair, immutable trace
scripts/
  run_agent.py   # offline scripted demo (--live for Azure OpenAI)
tests/
  test_llm.py        # decision parsing, ScriptedLLM, Azure guardrails
  test_tools.py      # calculator security, conversions, registry validation
  test_retrieval.py  # ranking, stopwords, edge cases
  test_agent.py      # multi-step runs, repair, failure capture, determinism
```

## Relation to the old repo

This supersedes `azure-langchain-agent` / `my-langchain-agent` (deprecated
`initialize_agent`/`load_tools` APIs, no tests, no tool validation). Same
product idea — an Azure-ready tool-using agent — rebuilt with a
framework-free core, tested end-to-end, and designed so the LLM and every
tool are replaceable without touching the loop.

## License

MIT
