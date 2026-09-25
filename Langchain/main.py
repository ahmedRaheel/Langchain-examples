from model import model

response = model.invoke(
    "Explain what an AI agent is in two sentences."
)

print(response.content)