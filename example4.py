from openai import OpenAI

OLLAMA_MODEL = "qwen2.5:0.5b"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",
)

print("=" * 40)
print("AI assistant — type quit or exit to stop")
print("=" * 40)

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in {"quit", "exit"}:
        print("Goodbye, have a nice day!")
        break

    if not user_input:
        continue

    response = client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "user", "content": user_input}
        ],
        temperature=0.2,
    )

    print("Assistant:")
    print(response.choices[0].message.content or "")