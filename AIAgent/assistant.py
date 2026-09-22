from openai import OpenAI
from planner import choose_tool
from tools import  get_current_time, generate_password, roll_dice

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "qwen2.5:0.5b"

client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

print("=" * 40)
print("AI assistant — type quit or exit to stop")
print("=" * 40)

while True:
    user_prompt = input("You:\n")

    if user_prompt.lower() in {"quit", "exit", "q", "e"}:
        print("Bye, Have a good day")
        break;

    if not user_prompt :
        continue
    tool = choose_tool(user_prompt)
    if tool == "get_current_time":    
        result = get_current_time()
    
    elif tool == "roll_dice":    
        result = roll_dice()
    
    elif tool == "generate_password":    
        result = generate_password()
    
    else:    
        result = None
    
    prompt = f"""
        The user asked:
        {user_prompt}
        The tool returned:
        {result}
        Answer the user naturally.don't add anything from your side in the answer. if tool is none, then answer it from llm model.
        """
    
    response = client.chat.completions.create(        
                model=OLLAMA_MODEL ,      
                messages=[
                    {
                    "role":"user",
                    "content":prompt
                }
                    ]
                    )
    print("\nAI :", response.choices[0].message.content)