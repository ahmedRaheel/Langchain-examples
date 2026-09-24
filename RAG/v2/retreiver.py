import numpy as np
from embeddings import create_embedding

def cosin_similarity(vector1, vector2):
    v1 = np.array(vector1)
    v2 = np.array(vector2)
    return np.dot(v1, v2) / (
            np.linalg.norm(v1)
            *
            np.linalg.norm(v2)
        )

def sementic_search(query, chunks, topk=5):
    embeded_query = create_embedding(query)
    results = []

    for chunk in chunks:
        score = cosin_similarity(embeded_query, 
                                 chunk["embedding"])

        results.append({
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "chunk_id": chunk["chunk_id"],
                    "semantic_score": score
                })

    results.sort(key=lambda item:
                 item["semantic_score"],
                   reverse= True)
    return results[:topk]

def keyword_search(query, text):

    query_words =  set(
        query.lower().split()
    )

    text_words = set(
            text.lower().split()
        )
    if not query_words:
        return 0.0
    matches = (
            query_words
            &
            text_words
        )
    
    return len(matches) / len(
            query_words
        )