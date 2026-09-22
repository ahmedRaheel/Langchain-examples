from openai import OpenAI as _OpenAI

OLLAMA_MODEL = "embeddinggemma:300m-qat-q4_0"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

client = _OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",
)

print("=" * 40)
print("Embedding implementation deep inside")
print("=" * 40)

user_input = input("Enter something to convert:")

response = client.embeddings.create(model= OLLAMA_MODEL, input= user_input)

print("Embedding:")
print(response.data[0].embedding or "")