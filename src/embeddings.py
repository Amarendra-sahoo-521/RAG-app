import streamlit as st
from sentence_transformers import SentenceTransformer

from src.config import DEFAULT_EMBEDDING_MODEL


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model_name)
