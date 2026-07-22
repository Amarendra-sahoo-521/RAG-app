import uuid

import chromadb
import streamlit as st


@st.cache_resource
def get_chroma_client():
    # In-memory client. Each uploaded PDF gets its own collection (see below),
    # so re-uploading never mixes documents from different files.
    return chromadb.Client()

def delete_collection(client, collection_name: str):
    """Remove a previously created collection so it doesn't leak in memory."""
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass  # already gone, or never existed — safe to ignore

def build_vector_store(chunks, embedder, client):
    collection_name = f"pdf_{uuid.uuid4().hex[:8]}"
    collection = client.create_collection(name=collection_name)

    texts = [c.page_content for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()
    ids = [f"doc_{i}_{uuid.uuid4().hex[:6]}" for i in range(len(chunks))]
    metadatas = [dict(c.metadata) for c in chunks]

    collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
    return collection


def retrieve_context(query: str, collection, embedder, top_k: int = 4):
    query_embedding = embedder.encode([query]).tolist()
    result = collection.query(query_embeddings=query_embedding, n_results=top_k)
    docs = result["documents"][0] if result.get("documents") else []
    metas = result["metadatas"][0] if result.get("metadatas") else [{}] * len(docs)
    return list(zip(docs, metas))
