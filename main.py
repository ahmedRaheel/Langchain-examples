from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain.tools import tool

from langchain_tavily import TavilySearch

from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun


@tool
def calculator(expression: str) -> str:
    """
    Perform a basic mathematical calculation.

    Example:
        25 * 4 + 10
    """
    try:
        # For learning/demo only.
        # In production, use a safe math parser instead of eval.
        result = eval(
            expression,
            {"__builtins__": {}},
            {}
        )

        return str(result)

    except Exception as exception:
        return f"Calculation error: {exception}"


# -----------------------------
# Model
# -----------------------------

model = ChatOllama(
    model="qwen2.5:0.5b",   # or "mistral", "qwen2.5", etc.
    temperature=0
)


# -----------------------------
# Tools
# -----------------------------

tavily_search = TavilySearch(
    tavily_api_key = "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", 
    max_results=5,
    topic="general"
)

wikipedia = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        top_k_results=3,
        doc_content_chars_max=4000
    )
)

tools = [
    tavily_search,
    
    calculator
]


# -----------------------------
# Agent
# -----------------------------

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="""
You are a research and reasoning assistant.

Use the available tools when appropriate.

Rules:

1. Use Tavily for current or recent information.
2. Use Wikipedia for established background knowledge.
3. Use the calculator for mathematical calculations.
4. Do not invent tool results.
5. If current information is requested, prefer Tavily over Wikipedia.
6. Provide a concise final answer after completing tool calls.
"""
)


# -----------------------------
# Execute Agent
# -----------------------------

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Who founded Microsoft, what year was it founded, "
                    "and calculate how many years have passed since then?"
                )
            }
        ]
    }
)


print(result["messages"][-1].content)