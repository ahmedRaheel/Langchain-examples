from langchain.agents import create_agent

from model import model
from tools import (
    current_time,
    roll_dice,
    generate_password
)


agent = create_agent(
    model=model,
    tools=[
        current_time,
        roll_dice,
        generate_password
    ]
)


result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is the current time?"
            }
        ]
    }
)


print(result["messages"][-1].content)
