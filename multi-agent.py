"""
AutoGen Multi-Agent Code Generation Workflow
============================================
Agents:
  Planner  → breaks task into numbered subtasks
  Coder    → writes Python for each subtask
  Critic   → reviews code; outputs PASS or FAIL + reasons
  UserProxy→ executes code in a sandbox; human steps in at TERMINATE

LLM backend: Ollama (OpenAI-compatible endpoint)
  ollama serve
  ollama pull llama3.2     # or any model you prefer

Install:
  pip install "autogen-agentchat==0.2.40"
"""

import os
import autogen

# ─────────────────────────────────────────────────────────────
# 1. Ollama LLM config
#    AutoGen's OpenAIWrapper honours base_url → points at Ollama
# ─────────────────────────────────────────────────────────────
OLLAMA_MODEL    = "qwen2.5:0.5b"          # change to any pulled model
OLLAMA_BASE_URL = "http://localhost:11434/v1"

# config_list is the standard AutoGen multi-model config format.
# Add more entries to fall back to other models on failure.
config_list = [
    {
        "model":    OLLAMA_MODEL,
        "base_url": OLLAMA_BASE_URL,
        "api_key":  "ollama",          # required by SDK; Ollama ignores value
    }
]

# Shared llm_config block reused by every AssistantAgent
llm_config = {
    "config_list":   config_list,
    "temperature":   0.2,              # low → deterministic code output
    "timeout":       120,              # seconds per LLM call
    "cache_seed":    None,             # set an int (e.g. 42) to cache responses
}

# ─────────────────────────────────────────────────────────────
# 2. Termination helper
#    Conversation ends when Critic outputs APPROVED or
#    UserProxy receives TERMINATE from any agent.
# ─────────────────────────────────────────────────────────────
def is_termination_msg(msg: dict) -> bool:
    """Return True when the workflow should stop."""
    content = msg.get("content", "") or ""
    return any(kw in content.upper() for kw in ["TERMINATE", "APPROVED", "TASK_COMPLETE"])


# ─────────────────────────────────────────────────────────────
# 3. Agent definitions
# ─────────────────────────────────────────────────────────────

planner = autogen.AssistantAgent(
    name="Planner",
    llm_config=llm_config,
    system_message="""You are a senior engineering lead.
Your ONLY job is to decompose the user's task into a clear, numbered plan.
Rules:
- Output a numbered list of subtasks, nothing else.
- Do NOT write any code — ever.
- Keep each subtask to one sentence.
- End your message with: "Coder, please implement the plan above."
""",
    description="Breaks tasks into subtasks. Never writes code.",
)

coder = autogen.AssistantAgent(
    name="Coder",
    llm_config=llm_config,
    system_message="""You are an expert Python engineer.
Your ONLY job is to write production-quality Python code.
Rules:
- Always include type hints.
- Always include a docstring.
- Always include error handling (try/except where appropriate).
- Wrap all code in a single python code block (```python ... ```).
- After the code block write: "Critic, please review the code above."
- Do NOT plan, do NOT review — only code.
""",
    description="Writes Python code only. Never plans or reviews.",
)

critic = autogen.AssistantAgent(
    name="Critic",
    llm_config=llm_config,
    system_message="""You are a senior code reviewer and security auditor.
Your ONLY job is to review the code written by Coder.
Review checklist:
  1. Correctness — does it solve the stated task?
  2. Type hints — are they present and accurate?
  3. Docstrings — present and informative?
  4. Error handling — are edge cases covered?
  5. Security — any injection, hardcoded secrets, unsafe exec()?
  6. Style — PEP 8 compliant?

Output format (use EXACTLY):
  VERDICT: PASS   ← if all 6 checks pass
  VERDICT: FAIL   ← if any check fails

Then list the reasons / improvements needed.

If PASS, end with: "APPROVED — UserProxy, you may execute the code."
If FAIL, end with: "Coder, please fix the issues above."
""",
    description="Reviews code quality and security. Outputs PASS or FAIL.",
)

# UserProxy runs code in a sandboxed directory and surfaces it to the human
# at TERMINATE.  human_input_mode="TERMINATE" means the human only types
# when an agent sends the termination signal — otherwise fully automated.
user_proxy = autogen.UserProxyAgent(
    name="UserProxy",
    human_input_mode="TERMINATE",       # prompt human only at end
    max_consecutive_auto_reply=8,       # hard ceiling — prevents infinite loops
    is_termination_msg=is_termination_msg,
    code_execution_config={
        "work_dir":        "sandbox",   # isolated working directory
        "use_docker":      False,       # set True for full sandbox isolation
        "timeout":         30,          # kill runaway code after 30 s
        "last_n_messages": 3,           # only look at last 3 msgs for code
    },
    default_auto_reply=(
        "Code executed. Please continue or type TERMINATE if done."
    ),
    system_message="A human proxy that executes code and escalates to a human at completion.",
)


# ─────────────────────────────────────────────────────────────
# 4. Speaker transition graph
#    Enforces the strict conversation flow:
#      UserProxy → Planner → Coder → Critic → Coder (on FAIL)
#                                           → UserProxy (on PASS)
#    Prevents agents from speaking out of turn.
# ─────────────────────────────────────────────────────────────
allowed_transitions = {
    user_proxy: [planner],          # kick-off goes to Planner
    planner:    [coder],            # Planner hands off to Coder
    coder:      [critic],           # Coder always goes to Critic
    critic:     [coder, user_proxy],# Critic → Coder (FAIL) or UserProxy (PASS)
}


# ─────────────────────────────────────────────────────────────
# 5. GroupChat + Manager
# ─────────────────────────────────────────────────────────────
groupchat = autogen.GroupChat(
    agents=[user_proxy, planner, coder, critic],
    messages=[],
    max_round=20,                   # absolute ceiling on total conversation turns
    speaker_selection_method="auto",# LLM-driven speaker selection
    allowed_or_disallowed_speaker_transitions=allowed_transitions,
    speaker_transitions_type="allowed",  # whitelist mode
    send_introductions=False,       # skip role-intro round (saves tokens)
)

manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config,
    # Manager itself uses the LLM only to resolve ambiguous speaker selection;
    # keep its system message minimal.
    system_message=(
        "You manage the group chat. Follow the allowed speaker transitions strictly."
    ),
    silent=False,                   # set True to suppress manager chatter
)


# ─────────────────────────────────────────────────────────────
# 6. Entrypoint
# ─────────────────────────────────────────────────────────────
def run(task: str) -> autogen.ChatResult:
    """
    Kick off the multi-agent workflow for a given task.

    Args:
        task: Natural language description of what to build.

    Returns:
        ChatResult with full conversation history and usage stats.

    Example:
        result = run("Write a Python function that merges two sorted lists.")
        print(result.summary)
    """
    # Create sandbox directory for code execution
    os.makedirs("sandbox", exist_ok=True)

    print("\n" + "=" * 65)
    print("AutoGen Multi-Agent Workflow — Ollama backend")
    print(f"Model : {OLLAMA_MODEL}  |  Max rounds : {groupchat.max_round}")
    print(f"Task  : {task}")
    print("=" * 65 + "\n")

    result = user_proxy.initiate_chat(
        recipient=manager,
        message=task,
        clear_history=True,          # fresh conversation each run
    )

    # ── Post-run summary ──────────────────────────────────────
    print("\n" + "=" * 65)
    print("WORKFLOW COMPLETE")
    print("=" * 65)

    total_rounds = len(groupchat.messages)
    print(f"Total turns : {total_rounds}")

    # Print the last Critic verdict
    for msg in reversed(groupchat.messages):
        if msg.get("name") == "Critic":
            print(f"Final verdict:\n{msg['content'][:400]}")
            break

    # Token usage (Ollama may return zeros — depends on model)
    if hasattr(result, "cost") and result.cost:
        print(f"Estimated cost : {result.cost}")

    return result


# ─────────────────────────────────────────────────────────────
# 7. Run
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run(
        task=(
            "Write a Python function called `merge_sorted_lists` that merges "
            "two sorted lists into a single sorted list without using built-in sort(). "
            "Include type hints, docstring, edge-case handling, and a few usage examples."
        )
    )