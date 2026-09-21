"""
Multi-Agent Code Generation Workflow — LangGraph v1.2+

Pipeline:
  research → code_gen → review → (retry loop or human_check) → END

Features:
  - Retry loop with escalation after 3 failed reviews
  - SQLite checkpointing (resume from any step after crash)
  - Human-in-the-loop interrupt before escalation
  - Simulated LLM nodes (swap in real calls by replacing _call_llm())
"""

import json
import sqlite3
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver


# ─────────────────────────────────────────────
# State schema
# ─────────────────────────────────────────────

class WorkflowState(TypedDict):
    task: str                   # Original user task
    research_result: str        # Output from research node
    code: str                   # Generated code
    review_passed: bool         # Did review pass?
    review_feedback: str        # Reviewer's comments
    iteration_count: int        # Retry counter (guard against infinite loops)
    human_approved: bool        # Set by human_checkpoint node
    final_output: str           # Assembled final result


# ─────────────────────────────────────────────
# Simulated LLM helper
# (Replace with real LLM calls — OpenAI, Anthropic, etc.)
# ─────────────────────────────────────────────

"""
Ollama configuration — edit these two values to match your setup.

  OLLAMA_MODEL : any model you have pulled locally, e.g.
                 "llama3.2", "mistral", "codellama", "deepseek-coder"
  OLLAMA_BASE_URL : default is localhost:11434; change if Ollama runs
                    on a remote host or a different port.

Pull a model first (run in your terminal):
    ollama pull llama3.2
"""
OLLAMA_MODEL    = "qwen2.5:0.5b"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

from openai import OpenAI as _OpenAI

# Ollama speaks the OpenAI chat-completions protocol.
# api_key can be any non-empty string — Ollama ignores it.
_ollama_client = _OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",
)


def _call_llm(system: str, user: str) -> str:
    """
    Call the locally-running Ollama model via its OpenAI-compatible endpoint.

    Ollama must be running before you execute this script:
        ollama serve          # starts the server
        ollama pull llama3.2  # download the model once
    """
    resp = _ollama_client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.2,   # low temp → more deterministic code output
    )
    return resp.choices[0].message.content


# ─────────────────────────────────────────────
# Node implementations
# ─────────────────────────────────────────────

def research_node(state: WorkflowState) -> dict:
    """
    Research node: gathers context and background for the task.
    Writes to: research_result
    """
    print(f"\n[RESEARCH] Researching task: '{state['task']}'")

    result = _call_llm(
        system="You are a senior research analyst. Summarise relevant approaches, data structures, and edge cases for the given task.",
        user=state["task"]
    )

    print(f"[RESEARCH] Done. {len(result)} chars.")
    return {"research_result": result}


def code_gen_node(state: WorkflowState) -> dict:
    """
    Code generation node: produces Python code based on research.
    Writes to: code, iteration_count
    """
    iteration = state["iteration_count"]
    print(f"\n[CODE GEN] Iteration {iteration}. Generating code...")

    prompt = (
        f"TASK:\n{state['task']}\n\n"
        f"RESEARCH:\n{state['research_result']}\n\n"
    )
    if iteration > 0 and state.get("review_feedback"):
        prompt += f"PREVIOUS REVIEW FEEDBACK (fix these issues):\n{state['review_feedback']}\n"

    code = _call_llm(
        system="You are an expert Python engineer. Write clean, production-ready Python code with type hints, docstrings, and error handling.",
        user=prompt
    )

    print(f"[CODE GEN] Done. {len(code)} chars.")
    return {
        "code": code,
        "iteration_count": iteration + 1,
    }


def review_node(state: WorkflowState) -> dict:
    """
    Review / QA node: evaluates the generated code.
    Writes to: review_passed, review_feedback
    """
    print(f"\n[REVIEW] Reviewing code (iteration {state['iteration_count']})...")

    prompt = (
        f"TASK:\n{state['task']}\n\n"
        f"CODE TO REVIEW:\n{state['code']}\n\n"
        f"ITERATION: {state['iteration_count']}\n"
        "Return JSON: {\"passed\": bool, \"feedback\": str}"
    )

    raw = _call_llm(
        system=(
            "You are a senior code reviewer. Check for: correctness, type hints, "
            "docstrings, error handling, edge cases. "
            "Return ONLY valid JSON: {\"passed\": bool, \"feedback\": str}"
        ),
        user=prompt
    )

    try:
        result = json.loads(raw)
        passed = bool(result.get("passed", False))
        feedback = str(result.get("feedback", ""))
    except json.JSONDecodeError:
        # Defensive: if LLM doesn't return JSON, fail safe
        passed = False
        feedback = f"Review node could not parse LLM output: {raw[:200]}"

    status = "✅ PASSED" if passed else "❌ FAILED"
    print(f"[REVIEW] {status} — {feedback[:80]}")
    return {"review_passed": passed, "review_feedback": feedback}


def human_checkpoint(state: WorkflowState) -> dict:
    """
    Human-in-the-loop node: pauses for human approval.
    In LangGraph, use interrupt() for async pause.
    Here we use a simple CLI prompt (works with checkpointer).
    Writes to: human_approved
    """
    print("\n" + "="*60)
    print("[HUMAN CHECK] Agent escalated — manual review required.")
    print("="*60)
    print(f"\nTASK:\n{state['task']}\n")
    print(f"ITERATIONS ATTEMPTED: {state['iteration_count']}")
    print(f"LAST REVIEW FEEDBACK:\n{state.get('review_feedback', 'N/A')}\n")
    print(f"GENERATED CODE:\n{state['code']}\n")
    print("="*60)

    while True:
        decision = input("\nApprove this output? [y]es / [n]o / [r]etry: ").strip().lower()
        if decision in ("y", "yes"):
            print("[HUMAN CHECK] Approved.")
            return {"human_approved": True}
        elif decision in ("n", "no"):
            print("[HUMAN CHECK] Rejected. Workflow will end without output.")
            return {"human_approved": False}
        elif decision in ("r", "retry"):
            # Reset iteration count to allow more retries
            print("[HUMAN CHECK] Retrying from code_gen...")
            return {
                "human_approved": False,
                "iteration_count": 0,
                "review_passed": False,
            }
        else:
            print("Please enter y, n, or r.")


def finalize_node(state: WorkflowState) -> dict:
    """
    Terminal node: assembles the final output artifact.
    Writes to: final_output
    """
    print("\n[FINALIZE] Assembling final output...")
    output = (
        f"# Task\n{state['task']}\n\n"
        f"# Research Summary\n{state['research_result']}\n\n"
        f"# Generated Code\n{state['code']}\n\n"
        f"# Review Status\n{'✅ Passed' if state['review_passed'] else '⚠️ Human approved'}\n"
        f"# Feedback\n{state.get('review_feedback', '')}\n"
        f"# Iterations\n{state['iteration_count']}\n"
    )
    return {"final_output": output}


# ─────────────────────────────────────────────
# Routing logic
# ─────────────────────────────────────────────

def should_retry(state: WorkflowState) -> Literal["human_check", "code_gen", "finalize"]:
    """
    After review:
    - review passed          → finalize
    - iteration_count > 3   → escalate to human
    - otherwise             → retry code_gen
    """
    if state["review_passed"]:
        return "finalize"
    if state["iteration_count"] > 3:
        print(f"\n[ROUTER] Max retries reached ({state['iteration_count']}). Escalating to human.")
        return "human_check"
    print(f"\n[ROUTER] Review failed. Retrying (attempt {state['iteration_count'] + 1}/3).")
    return "code_gen"


def after_human_check(state: WorkflowState) -> Literal["finalize", "code_gen", "__end__"]:
    """
    After human review:
    - approved            → finalize
    - retry (count reset) → code_gen
    - rejected            → END
    """
    if state["human_approved"]:
        return "finalize"
    if state["iteration_count"] == 0:
        # Human chose retry (we reset iteration_count to 0 in human_checkpoint)
        return "code_gen"
    return END


# ─────────────────────────────────────────────
# Graph assembly
# ─────────────────────────────────────────────

def build_graph(checkpointer) -> StateGraph:
    graph = StateGraph(WorkflowState)

    # Register nodes
    graph.add_node("research",      research_node)
    graph.add_node("code_gen",      code_gen_node)
    graph.add_node("review",        review_node)
    graph.add_node("human_check",   human_checkpoint)
    graph.add_node("finalize",      finalize_node)

    # Entry point
    graph.set_entry_point("research")

    # Linear edges
    graph.add_edge("research", "code_gen")
    graph.add_edge("code_gen", "review")

    # Conditional edges
    graph.add_conditional_edges(
        "review",
        should_retry,
        {
            "code_gen":    "code_gen",
            "human_check": "human_check",
            "finalize":    "finalize",
        }
    )

    graph.add_conditional_edges(
        "human_check",
        after_human_check,
        {
            "finalize":  "finalize",
            "code_gen":  "code_gen",
            END:         END,
        }
    )

    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def run(task: str, thread_id: str = "default") -> str:
    """
    Run the workflow for a given task.

    Args:
        task:      Natural language description of what to build.
        thread_id: Unique ID for this run — enables checkpoint resume.

    Returns:
        The final assembled output string.
    """
    db_path = "./state.db"
    config = {"configurable": {"thread_id": thread_id}}

    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        app = build_graph(checkpointer)

        initial_state: WorkflowState = {
            "task": task,
            "research_result": "",
            "code": "",
            "review_passed": False,
            "review_feedback": "",
            "iteration_count": 0,
            "human_approved": False,
            "final_output": "",
        }

        print(f"\n{'='*60}")
        print(f"Starting workflow — thread: {thread_id}")
        print(f"Task: {task}")
        print(f"{'='*60}")

        final_state = app.invoke(initial_state, config=config)

        print(f"\n{'='*60}")
        print("WORKFLOW COMPLETE")
        print(f"{'='*60}")
        print(final_state.get("final_output", "No output produced."))

        return final_state.get("final_output", "")


# ─────────────────────────────────────────────
# How to resume a crashed / interrupted run:
#
#   with SqliteSaver.from_conn_string("./state.db") as checkpointer:
#       app = build_graph(checkpointer)
#       # Pass the same thread_id — LangGraph replays from last checkpoint
#       app.invoke(None, config={"configurable": {"thread_id": "my-run-001"}})
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run(
        task="Write a Python function that factorial of numbers 1 to 10.",
        thread_id="demo-run-001",
    )