from langchain_groq import ChatGroq


def build_prompt(query: str, context_chunks) -> str:
    context = "\n\n".join(text for text, _ in context_chunks)
    return f"""Use the context below to answer the question.
If the answer is not contained in the context, say you don't know instead of guessing.

Context:
{context}

Question: {query}
Answer:"""


def get_llm(api_key: str, model: str):
    return ChatGroq(
        groq_api_key=api_key,
        model=model,
        temperature=0.1,
        max_tokens=1024,
    )
