from openai import OpenAI as _OpenAI

OLLAMA_MODEL = "qwen2.5:0.5b"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

client = _OpenAI(base_url= OLLAMA_BASE_URL, api_key="ollama")
roles = {"1": "You are expert .net core developer ", 
        "2":"You are friendly cricket expert", 
         "3": "You are friendly math teacher for primary classses"
        }

print("=" * 40)
print("AI personal assistent")
print("=" * 40)

print("\nChoose Your Assistant\n")
print("1 .net expert")
print("2. Cricket expert")
print("3. Math expert")

choice = input("\nEnter your choice : ")
messages = [{"role":"system", 
             "content": roles.get(choice , "You are Helpful assistant")

               }]
while True:
    user_input = input("You:\n")

    if user_input.lower() in {"quit", "exit"}:
        print("Bye! Have a nice day")
        break

    if not user_input:
        continue

    messages.append  ({"role": "user", "content": user_input})

    response = client.chat.completions.create(model= OLLAMA_MODEL, 
                                              temperature=0.2, 
                                              messages= messages)  
    assistant_reply = response.choices[0].message.content

    messages.append ({"role": "assistant", "content": assistant_reply})

    print("Assistant: \n")
    print(assistant_reply)
