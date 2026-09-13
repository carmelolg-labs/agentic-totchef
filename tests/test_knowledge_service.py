"""
Unit tests for lib.core.service.KnowledgeService.

The Ollama embedding calls are mocked with deterministic one-hot vectors so
Chroma's cosine similarity math is exact and reproducible, with no network
call to a real Ollama server.
"""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import lib.core.service.KnowledgeService as ks


class TestGetBestMatchingChunk:
    def test_returns_best_match_with_similarity(self):
        chunks = ["apple", "banana", "cherry"]
        doc_vectors = {
            "apple": [1.0, 0.0, 0.0],
            "banana": [0.0, 1.0, 0.0],
            "cherry": [0.0, 0.0, 1.0],
        }

        with patch.object(ks, "_embeddings") as mock_embeddings:
            mock_embeddings.embed_documents.side_effect = (
                lambda texts: [doc_vectors[t] for t in texts]
            )
            mock_embeddings.embed_query.return_value = doc_vectors["apple"]

            result = ks.get_best_matching_chunk("apple-ish query", chunks)

        assert result is not None
        assert result["match"] == "apple"
        assert result["similarity"] == 1.0

    def test_returns_none_when_chunks_empty(self):
        result = ks.get_best_matching_chunk("query", [])
        assert result is None

    def test_returns_none_when_query_empty(self):
        result = ks.get_best_matching_chunk("", ["apple", "banana"])
        assert result is None

    def test_low_similarity_still_returns_closest_match(self):
        chunks = ["carbohydrates", "proteins", "vegetables"]
        doc_vectors = {
            "carbohydrates": [1.0, 0.0, 0.0],
            "proteins": [0.0, 1.0, 0.0],
            "vegetables": [0.0, 0.0, 1.0],
        }
        # query orthogonal to "vegetables" but closest (45 degrees) to "carbohydrates"
        query_vector = [0.5, 0.0, 0.5]

        with patch.object(ks, "_embeddings") as mock_embeddings:
            mock_embeddings.embed_documents.side_effect = (
                lambda texts: [doc_vectors[t] for t in texts]
            )
            mock_embeddings.embed_query.return_value = query_vector

            result = ks.get_best_matching_chunk("something unrelated", chunks)

        assert result is not None
        assert result["match"] in ("carbohydrates", "vegetables")
        assert 0.0 < result["similarity"] < 1.0

    def test_concurrent_calls_do_not_race(self):
        """Regression test for the _chroma_lock: several threads hammering
        get_best_matching_chunk at once (mirroring LangGraph's ToolNode
        dispatching multiple tool calls in parallel within one agent turn)
        must not raise, since Chroma's client construction/teardown is not
        thread-safe without the lock.
        """
        chunks = ["carbohydrates", "proteins", "vegetables"]
        doc_vectors = {
            "carbohydrates": [1.0, 0.0, 0.0],
            "proteins": [0.0, 1.0, 0.0],
            "vegetables": [0.0, 0.0, 1.0],
        }

        with patch.object(ks, "_embeddings") as mock_embeddings:
            mock_embeddings.embed_documents.side_effect = (
                lambda texts: [doc_vectors[t] for t in texts]
            )
            mock_embeddings.embed_query.return_value = doc_vectors["proteins"]

            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(
                    lambda _: ks.get_best_matching_chunk("protein-ish", chunks),
                    range(16),
                ))

        assert all(r is not None and r["match"] == "proteins" for r in results)
