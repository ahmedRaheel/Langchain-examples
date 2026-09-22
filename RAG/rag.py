from openai import OpenAI
from retiever import get_all_documents
from tools import cosine_similarity

OLLAMA_EMBEDDING_MODEL = "embeddinggemma:300m-qat-q4_0"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "qwen2.5:0.5b"

client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",
)

documents = get_all_documents()
documents_embeddings = {}

for filename, content in documents.items():
    file_embeding = client.embeddings.create(
            model= OLLAMA_EMBEDDING_MODEL,
            input=content
     ).data[0].embedding
    documents_embeddings[filename] = file_embeding

print("=" * 40)
print("      My AI Assistant")
print("=" * 40)

while True:
    user_input = input("\nYou : ")
    
    if user_input.lower() == "quit":
        print("\nAI  : Goodbye! Have a great day.")
        break
    embedding_question =  client.embeddings.create(
            model= OLLAMA_EMBEDDING_MODEL,
            input=user_input
     ).data[0].embedding

    best_score = -1
    
    best_document = None

    for filename, embedding in documents_embeddings.items():
        score = cosine_similarity(embedding_question, embedding)

        if score > best_score:
            best_score = score
            best_document = filename
    context = documents[best_document]

    prompt = f"""
    Answer the question using only
    the following information.
    
    Context:
    
    {context}
    
    Question:
    
    {user_input}
    """
    
    
    response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
    
    print("\nAI :", response.choices[0].message.content)