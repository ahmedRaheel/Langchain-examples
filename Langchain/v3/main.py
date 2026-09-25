from model import model
from prompt import prompt
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

chain = (
    prompt
    | model
    | parser
)

response = chain.invoke(
    {
        "question": "Why is MCP useful for AI agents?"
    }
)

print(response)
