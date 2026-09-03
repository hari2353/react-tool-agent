"""Run the tool-calling agent demo - no credentials required.

Uses ScriptedLLM (a deterministic decision replay) so the exact multi-step
behavior below runs identically on every machine and in CI:

1. the agent searches the knowledge base for audit-cadence policy,
2. converts 100 miles to km (unit conversion tool),
3. computes 36 * 12 with the calculator,
4. produces a final grounded answer.

Set Azure credentials (AZURE_OPENAI_*) and pass --live to run the same
loop against a real Azure OpenAI deployment instead.

Usage:
    python scripts/run_agent.py [--live]
"""

from __future__ import annotations

import argparse
import json

from tool_agent.agent import AgentConfig, ToolAgent
from tool_agent.llm import AzureOpenAILLM, ScriptedLLM
from tool_agent.retrieval import default_store
from tool_agent.tools import default_registry

_SCRIPT = [
    json.dumps(
        {
            "action": "tool",
            "tool": "kb_search",
            "arguments": {"query": "audit cadence by risk band", "top_k": 2},
        }
    ),
    json.dumps(
        {
            "action": "tool",
            "tool": "unit_convert",
            "arguments": {"value": 100, "unit": "mi", "target": "km"},
        }
    ),
    json.dumps(
        {"action": "tool", "tool": "calculator", "arguments": {"expression": "36 * 12"}}
    ),
    json.dumps(
        {
            "action": "final",
            "answer": "Audit cadence: high-risk suppliers are re-audited every "
            "12 months, medium-risk every 24 months, and low-risk suppliers "
            "complete a desktop self-assessment every 36 months (432 weeks "
            "apart); any supplier with an open critical corrective action is "
            "re-audited within 6 months regardless of band. For context: "
            "100 miles is about 160.93 km.",
        }
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true",
                        help="use Azure OpenAI instead of the scripted LLM")
    args = parser.parse_args()

    store = default_store()
    registry = default_registry(store)

    if args.live:
        llm = AzureOpenAILLM()
        print(f"LLM: Azure OpenAI deployment={llm._deployment}")
    else:
        llm = ScriptedLLM(script=_SCRIPT)
        print("LLM: ScriptedLLM (deterministic replay - no credentials needed)")

    agent = ToolAgent(llm=llm, tools=registry, config=AgentConfig(max_steps=8))

    questions = [
        "How often are high-risk suppliers audited, and how far apart are "
        "low-risk self-assessments in weeks? Also convert 100 miles to km "
        "for a site-visit distance estimate.",
    ]
    for question in questions:
        print()
        print("=" * 72)
        print(f"QUESTION: {question}")
        print("=" * 72)
        result = agent.run(question)
        for step in result.steps:
            if step.kind == "tool_call":
                print(f"  step {step.step}: tool: {step.tool}({step.detail})")
            elif step.kind == "tool_result":
                print(f"  step {step.step}: -> {step.detail}")
            elif step.kind == "final":
                print(f"  step {step.step}: final answer")
            else:
                print(f"  step {step.step}: {step.kind}: {step.detail}")
        print()
        print(f"SUCCESS   : {result.success}")
        print(f"TOOLS USED: {result.tools_used}")
        print(f"STEPS     : {result.n_steps}")
        print()
        print("FINAL ANSWER:")
        print(f"  {result.answer}")

    if not args.live:
        scripted = ScriptedLLM(script=_SCRIPT)
        assert scripted.remaining == len(_SCRIPT)
    print()
    print("demo complete")


if __name__ == "__main__":
    main()
