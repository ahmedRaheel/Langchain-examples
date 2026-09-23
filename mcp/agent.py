import asyncio
from openai import OpenAI
from mcp_client import (
    connect,
    disconnect,
    discover_tools,
    execute_tool
)

# Load configuration

OLLAMA_MODEL = "qwen2.5:0.5b"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

# Create AI client
client = OpenAI(
    base_url= OLLAMA_BASE_URL,
    api_key= "ollama"
)


# Build Tool Descriptions
def build_tool_descriptions(tools):

    descriptions = ""
    for tool in tools:
        descriptions += f"""
        Tool:
        {tool.name}

        Description:
        {tool.description}
        ----------------------------
        """

    return descriptions

def planner(user_request, tools):
    tool_description = build_tool_descriptions(tools)
    ###Then create a prompt by combining user input and Tool Description.  and Give this prompt to Ollama. ###
    prompt = f"""
    You are an AI planner.
    Avaiable tools:
    {tool_description}
    1. Select the best tool
    2. Reply only with tool name
    3. Donot explain 
    User Request:
    {user_request}
          """
    response = client.chat.completions.create(model=OLLAMA_MODEL,
                                               messages= [{"role":"user", "content": prompt}],
                                                temperature=0.2)

    tool_name = response.choices[0].message.content

    return tool_name.strip()

async def main():
    mcp_client = await connect()
    tools = await discover_tools(mcp_client)

    user_input = input("You:\n")
    tool_name = planner(user_input, tools)
    print("Planner Selected:",
        tool_name)

   

asyncio.run(main())