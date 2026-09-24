from openai import OpenAI
OLLAMA_MODEL = "embeddinggemma:300m-qat-q4_0"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",
)

def create_embedding(source_text):
    response = client.embeddings.create(model= OLLAMA_MODEL,
                                         input= source_text)
    return response.data[0].embedding or ""

def embed_chunks(chunks):
    embedded_chunks = []
    
    for chunk in chunks:
    
            embedding = create_embedding(
                chunk["text"]
            )
    
            embedded_chunks.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
                "embedding": embedding
            })
    
    return embedded_chunks
    
    
