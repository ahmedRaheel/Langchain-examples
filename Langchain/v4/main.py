import asyncio

from langchain.agents import create_agent

from model import model
from mcp_tools import client


async def main():
    tools = await client.get_tools()

    agent = create_agent(
        model=model,
        tools=tools
    )

    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is the current time?"
                }
            ]
        }
    )

    print("\nFINAL ANSWER")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())

