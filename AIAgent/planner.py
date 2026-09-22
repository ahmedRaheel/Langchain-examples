from openai import OpenAI
OLLAMA_EMBEDDING_MODEL = "embeddinggemma:300m-qat-q4_0"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "qwen2.5:0.5b"

client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
def choose_tool(user_request):

    planner_prompt = f"""
You are an AI planner.

Available tools:

1. get_current_time
   Use when the user asks for the current date or time.

2. roll_dice
   Use when the user asks to roll a dice.

3. generate_password
   Use when the user wants a secure password.

If no tool is required, return:

none

Return ONLY the tool name.

User Request:

{user_request}
"""

    response = client.chat.completions.create(

        model=OLLAMA_MODEL,
        messages=[
            {
                "role":"system",
                "content":"You are an AI planner."
            },
            {
                "role":"user",
                "content":planner_prompt
            }
        ]
    )

    return response.choices[0].message.content.strip()


