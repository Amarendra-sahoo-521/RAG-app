import streamlit as st

from src.config import DEFAULT_GROQ_MODEL, DEFAULT_TOP_K
from src.embeddings import load_embedder
from src.llm import build_prompt, get_llm
from src.pdf_processor import load_and_split_pdf
from src.vector_store import build_vector_store, get_chroma_client, retrieve_context,delete_collection

st.set_page_config(page_title="Chat with your PDF", page_icon="📄", layout="wide")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "collection" not in st.session_state:
    st.session_state.collection = None
if "processed_file_names" not in st.session_state:
    st.session_state.processed_file_names = None

# ---------------------------------------------------------------------------
# Sidebar: theme + config + upload
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Setup")
    st.divider()

    groq_api_key = st.text_input(
        "API key",
        type="password",
        help="Get one at https://console.groq.com/keys. Never hardcode this in your code.",
    )
    model_name = st.text_input("Model name", value=DEFAULT_GROQ_MODEL)

    st.divider()

    uploaded_files = st.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=True)
    top_k = st.slider("Chunks to retrieve (top-k)", min_value=1, max_value=10, value=DEFAULT_TOP_K)
    current_file_names = sorted(f.name for f in uploaded_files) if uploaded_files else []
    
    if uploaded_files and current_file_names != st.session_state.processed_file_names:
        with st.spinner("Reading, chunking, and embedding your {len(uploaded_files)} PDF(s)..."):
            embedder = load_embedder()
            client = get_chroma_client()
            
            if st.session_state.collection is not None:
                delete_collection(client, st.session_state.collection.name)
                
            all_chunks = []
            for f in uploaded_files:
                all_chunks.extend(load_and_split_pdf(f))
            
            collection = build_vector_store(all_chunks, embedder, client)

        st.session_state.collection = collection
        st.session_state.processed_file_names = current_file_names
        st.session_state.messages = []  # reset chat when a new PDF is loaded
        st.success(f"Indexed {len(all_chunks)} chunks from {len(uploaded_files)} file(s)")
        
    elif not uploaded_files and st.session_state.processed_file_names:
        # user cleared the uploader — clean up the old collection too
        client = get_chroma_client()
        if st.session_state.collection is not None:
            delete_collection(client, st.session_state.collection.name)
        st.session_state.collection = None
        st.session_state.processed_file_names = []
        st.session_state.messages = []
        
    if st.session_state.processed_file_names:
        st.caption(f"Active document: **{st.session_state.processed_file_names}**")


# ---------------------------------------------------------------------------
# Main: chat interface
# ---------------------------------------------------------------------------
st.title("📄 Chat with your PDF")

if not st.session_state.collection:
    st.info("Upload a PDF from the sidebar to get started.")
    st.stop()

if not groq_api_key:
    st.warning("Enter your API key in the sidebar to ask questions.")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about your PDF...")

if question:
    if not groq_api_key:
        st.error("Please enter an API key in the sidebar first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            embedder = load_embedder()
            
            # semantic search
            context_chunks = retrieve_context(
                question, st.session_state.collection, embedder, top_k=top_k
            )

            if not context_chunks:
                answer = "I couldn't find anything relevant to that in the document."
            else:
                prompt = build_prompt(question, context_chunks)
                llm = get_llm(groq_api_key, model_name)
                response = llm.invoke([prompt])
                answer = response.content

            st.markdown(answer)

            with st.expander("Sources used"):
                for text, meta in context_chunks:
                    source = meta.get("source", "unknown")
                    page = meta.get("page_label", meta.get("page", "?"))
                    st.markdown(f"**{source}, page {page}:** {text[:300]}...")

    st.session_state.messages.append({"role": "assistant", "content": answer})
