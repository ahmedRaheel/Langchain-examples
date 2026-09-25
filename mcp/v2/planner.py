from openai import OpenAI
import json


OLLAMA_MODEL = "qwen2.5:0.5b"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",
)


def build_tool_descriptions(tools):

    descriptions = ""

    for tool in tools:

        descriptions += f"""
Tool:
{tool.name}

Description:
{tool.description}

Input Schema:
{tool.inputSchema}

----------------------------
"""

    return descriptions


def planner(
    state,
    tools
):

    tool_descriptions = (
        build_tool_descriptions(tools)
    )

    prompt = f"""
You are an AI Planner.

Your job is to decide ONLY the
next action.

User Request:

{state["user_request"]}

Completed Actions:

{state["actions"]}

Previous Observations:

{state["observations"]}

Available Tools:

{tool_descriptions}

Rules:

1. Choose only ONE next action.

2. Never repeat an action that has
   already been completed.

3. Use the observations to decide
   what is still required.

4. If the user's request has been
   completely satisfied, return FINISH.

5. Return ONLY valid JSON.

6. Do not explain your answer.

Use exactly this format:

{{
    "tool": "tool_name",
    "arguments": {{
        "argument_name": "argument_value"
    }}
}}

For FINISH, return:

{{
    "tool": "FINISH",
    "arguments": {{}}
}}

Next Action:
"""

    response = client.chat.completions.create(
        model= OLLAMA_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an AI planner."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    result = (
        response
        .choices[0]
        .message
        .content
        .strip()
    )

    return json.loads(result)

