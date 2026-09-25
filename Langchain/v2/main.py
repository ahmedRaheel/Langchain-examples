from model import model
from prompt import prompt

messages = prompt.invoke(
    {
        "question": "What is an autonomous AI agent?"
    }
)

response = model.invoke(messages)

print(response.content)

