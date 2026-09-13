"""
Knowledge Service Module
Provides a Chroma-backed semantic matching function to find the best matching
chunk of text for a given query, used for fuzzy category lookups.
"""

import threading
import uuid
from typing import Iterable, Optional, Dict, Any

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from lib.commons.EnvironmentVariables import get_embedding_model, get_ollama_host

_embeddings = OllamaEmbeddings(model=get_embedding_model(), base_url=get_ollama_host())

# chromadb's Rust-backed SharedSystemClient is not safe under concurrent
# construction/teardown: an agent turn that fires several tool calls at once
# (LangGraph's ToolNode dispatches those in parallel via its own thread pool)
# can hit "AttributeError: 'RustBindingsAPI' object has no attribute
# 'bindings'" when two threads build/release a chromadb.Client() at the same
# instant. Serializing access here is cheap (a few ms per call) and removes
# the race regardless of where the concurrency comes from.
_chroma_lock = threading.Lock()


def get_best_matching_chunk(query: str, chunks: Iterable[str]) -> Optional[Dict[str, Any]]:
    """Finds the best matching chunk from a list of chunks based on a query.

    Builds an ephemeral, in-memory Chroma collection from the given chunks
    (rebuilt on every call since the dataset is small and static) and returns
    the closest match by cosine similarity.

    Args:
        query (str): The input query string to find the best matching chunk for.
        chunks (Iterable[str]): An iterable of chunk strings.
    Returns:
        Optional[Dict[str, Any]]: A dictionary containing the best matching chunk
            and its cosine similarity score, or None if no match is found.
    """
    chunks = list(chunks)
    if not query or not chunks:
        return None

    # A unique collection_name per call keeps this isolated: chromadb's
    # default in-memory client shares one underlying store across every
    # Chroma instance in the process, so a fixed/default name would silently
    # accumulate chunks from every previous call instead of matching only
    # against `chunks`. The store itself is still shared, so the collection
    # is explicitly deleted afterwards — otherwise it (and its embedded
    # vectors) would leak in that shared store for the lifetime of the process.
    with _chroma_lock:
        store = Chroma.from_texts(
            texts=chunks,
            embedding=_embeddings,
            collection_name=str(uuid.uuid4()),
            collection_metadata={"hnsw:space": "cosine"},
        )
        try:
            results = store.similarity_search_with_score(query, k=1)
        finally:
            store.delete_collection()

    if not results:
        return None

    document, distance = results[0]
    similarity = 1 - distance

    return {"match": document.page_content, "similarity": similarity}
