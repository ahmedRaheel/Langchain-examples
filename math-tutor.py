import os
from openai import OpenAI as _OpenAI

OLLAMA_MODEL = "qwen2.5:0.5b"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

_client = _OpenAI( base_url= OLLAMA_BASE_URL, api_key="ollama")
print("=" * 40)
print("AI assistant for mathematics — type quit or exit to stop")
print("=" * 40)

messages = [{"role": "system",  "content": (
            "You are a Friend math tutor. "
            "Only answer questions about Math. "
            "Add real time examples"
           
        )}]

while True:
    user_input = input("You:\n")

    if user_input.lower() in {"quit", "exit"}:
        print("Good bye! Have a  nice day")
        break;
    if not user_input:
        continue

    messages.append({"role": "user", "content": user_input})

    response = _client.chat.completions.create(
        model= OLLAMA_MODEL, 
        temperature=0.2,
        messages= messages
    )

    assistant_reply = response.choices[0].message.content or ""
    messages.append({"role": "assistant", "content":  assistant_reply})
    print("Assistant:\n")
    print(assistant_reply)
