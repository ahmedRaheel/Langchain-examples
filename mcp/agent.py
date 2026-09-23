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

def generate_response(user_request, tool):
    prompt = f"""
      The user asked:
      {user_request}
      The tool suggested by planner 
      {tool}
      Respond directly to the user in a natural, conversational way.

Use the tool result as the factual source.
Transform raw tool output into a human-friendly answer.
Do not simply copy raw values when a natural sentence would be better.
Do not add information that is not needed to answer the request.
Do not mention tools, internal processing, planning, or reasoning.
Keep the response concise.
       """
    response = client.chat.completions.create(
        model= OLLAMA_MODEL, 
        messages= [{"role": "user", "content": prompt}]
    )
    reply = response.choices[0].message.content or ""
    return reply

async def main():
    mcp_client = await connect()
    tools = await discover_tools(mcp_client)

    user_input = input("You:\n")
    tool_name = planner(user_input, tools)
    print("Planner Selected:",
        tool_name)
    print()
    tool_result = await execute_tool(mcp_client, tool_name)
    answer = generate_response(user_input, tool_result)

    print("AI:\n")
    print(answer)

   

asyncio.run(main())