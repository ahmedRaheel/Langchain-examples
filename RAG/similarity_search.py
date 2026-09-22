import numpy as np
from openai import OpenAI 
OLLAMA_MODEL = "embeddinggemma:300m-qat-q4_0"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",
)


def cosine_similarity(vector1, vector2):

    v1 = np.array(vector1)
    v2 = np.array(vector2)

    return np.dot(v1, v2) / (
        np.linalg.norm(v1) *
        np.linalg.norm(v2)
    )


text1 = "Python is a programming language."

text2 = "Math is base of all subject"

embedding1 = client.embeddings.create(
    model= OLLAMA_MODEL,
    input=text1
).data[0].embedding

embedding2 = client.embeddings.create(
    model=OLLAMA_MODEL,
    input=text2
).data[0].embedding


print(len(embedding1))
print(len(embedding2))

score = cosine_similarity(
    embedding1,
    embedding2
)

print(score)

