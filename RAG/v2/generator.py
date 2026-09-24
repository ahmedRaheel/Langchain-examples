from openai import OpenAI


client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)


MODEL = "qwen2.5:0.5b"


def generate_answer(
    query,
    results
):

    context = "\n\n".join(
        result["text"]
        for result in results
    )

    prompt = f"""
Answer the user's question using
only the provided context.

Context:
{context}

Question:
{query}

Rules:

1. Use only the provided context.
2. Do not invent information.
3. If the answer is not present,
   say that the information is
   not available.
4. Give a clear and concise answer.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return (
        response
        .choices[0]
        .message
        .content
        .strip()
    )

