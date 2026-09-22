from tools import get_current_time
from openai import OpenAI as _OpenAI

OLLAMA_MODEL = "qwen2.5:0.5b"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

client = _OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",
)

print("=" * 40)
print("AI assistant — type quit or exit to stop")
print("=" * 40)

messages =  [{"role": "system", "content": "You are friend AI Assistent with custom tools"}]

while True:
    user_input = input("You\n")

    if not user_input:
        continue;
   

    if user_input in {"quit", "exit"}:
        print("Bye!, have a nice day")
        break;

    messages.append({"role":"user", "content":user_input})

    if "time"  in user_input:
            print("Assistant:\n")
            print(get_current_time())
            continue
    response = client.chat.completions.create(model= OLLAMA_MODEL, 
                                              temperature=0.2, 
                                              messages= messages)

    assistant_reply = response.choices[0].message.content or "";

    messages.append({"role":"assistant", "content": assistant_reply})
    print("Assistant:\n")
    print(assistant_reply)




