from chunker import create_chunk
from embeddings import (
    embed_chunks,
    create_embedding
)

from chroma_store import (
    add_chunk,
    search as chroma_search
)

from retreiver import keyword_search
from reranker import rerank


def build_rag(
    text,
    source
):

    chunks = create_chunk(
        text,
        source=source,
        chunk_size=200,
        overlap=50
    )

    embedded_chunks = embed_chunks(
        chunks
    )

    add_chunk(
        embedded_chunks
    )

    return embedded_chunks


def retrieve(
    query,
    top_k=5
):

    query_embedding = create_embedding(
        query
    )

    semantic_results = chroma_search(
        query_embedding,
        top_k=top_k
    )

    candidates = []

    documents = (
        semantic_results["documents"][0]
    )

    metadatas = (
        semantic_results["metadatas"][0]
    )

    distances = (
        semantic_results["distances"][0]
    )

    for i in range(len(documents)):

        text = documents[i]

        metadata = metadatas[i]

        distance = distances[i]

        semantic = 1 / (
            1 + distance
        )

        keyword = keyword_search(
            query,
            text
        )

        combined = (
            0.7 * semantic
            +
            0.3 * keyword
        )

        candidates.append({
            "text": text,
            "source": metadata["source"],
            "chunk_id": metadata["chunk_id"],
            "semantic_score": semantic,
            "keyword_score": keyword,
            "combined_score": combined
        })

    candidates.sort(
        key=lambda item:
        item["combined_score"],
        reverse=True
    )

    candidates = candidates[:top_k]
    print(                "\nRetrieved Sources Candidates:"
            )
    
    for candidate in candidates:
    
        print(
                    f"- "
                    f"{candidate['source']} "
                    f"(chunk "
                    f"{candidate['chunk_id']})"
                )
    
    return rerank(
        query,
        candidates,
        top_k=3
    )
