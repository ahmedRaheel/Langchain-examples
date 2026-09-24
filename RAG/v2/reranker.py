from openai import OpenAI
OLLAMA_MODEL = "qwen2.5:0.5b"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",
)

def rerank(query, candidates, topk=3):
    scored = []

    for candidate in candidates:
        prompt = f"""
You are a document relevance evaluator.
User request:
{query}
Document:
{candidate["text"]}

Give a relevance score from 0 to 100.

100 = directly answers the query
0 = completely irrelevant

Return ONLY the number.

"""
        response = client.chat.completions.create(
            model= OLLAMA_MODEL,
            messages=[{
                "role":"user", 
                "content": prompt
            }],
            temperature=0
        )

        score_text = (
                    response
                    .choices[0]
                    .message
                    .content
                    .strip()
                )
        try:
            score = float(score_text)
        except ValueError:
            score = 0
        scored.append({
                    **candidate,
                    "rerank_score": score
                })

    scored.sort(
            key=lambda item:
            item["rerank_score"],
            reverse=True
        )

    return scored[:topk]
