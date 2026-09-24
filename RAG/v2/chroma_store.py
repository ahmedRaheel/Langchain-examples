import chromadb

client = chromadb.PersistentClient(path="./rag_db")




collection = (
    client.get_or_create_collection(
        name="documents"
    )
)

def add_chunk(chunks):
    documents = []
    ids = []
    embeddings = []
    metadatas =  []
    for chunk in chunks:
        documents.append(
                    chunk["text"]
                )
        
        embeddings.append(
                    chunk["embedding"]
                )
        
        ids.append(
                    f'{chunk["source"]}_'
                    f'{chunk["chunk_id"]}'
                )
        
        metadatas.append({
                    "source": chunk["source"],
                    "chunk_id": chunk["chunk_id"]
                })

        collection.upsert(
             ids=ids,
             documents=documents,
             embeddings=embeddings,
             metadatas=metadatas
        )

def search(embedding_query, top_k = 5):
    """ sementic search"""

    return  collection.query(
        query_embeddings= [embedding_query],
          n_results= top_k)




