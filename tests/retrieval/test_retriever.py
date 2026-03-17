"""Tests for semantic retriever."""

import pytest
from unittest.mock import Mock, MagicMock
from src.retrieval.retriever import SemanticRetriever
from src.models.document import Document
from src.models.search_result import SearchResult
from src.models.code_entity import CodeEntity, EntityType


class TestSemanticRetriever:
    """Test suite for SemanticRetriever."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        store = Mock()
        return store

    @pytest.fixture
    def mock_embedder(self):
        """Create a mock embedder."""
        embedder = Mock()
        embedder.embed.return_value = [0.1] * 384
        return embedder

    @pytest.fixture
    def retriever(self, mock_vector_store, mock_embedder):
        """Create a retriever instance with mocks."""
        return SemanticRetriever(mock_vector_store, mock_embedder, default_limit=10)

    @pytest.fixture
    def sample_search_results(self):
        """Create sample search results for testing."""
        results = []

        # Document result
        doc1 = Document(
            id="doc1",
            content="This is a document about authentication",
            title="Authentication Guide",
            source="confluence://docs/auth",
            source_type="document",
            metadata={"space": "docs"},
        )
        results.append(SearchResult(document=doc1, score=0.95, rank=1))

        # Code result
        doc2 = Document(
            id="code1",
            content="def authenticate(user):\n    return validate(user)",
            title="authenticate",
            source="src/auth.py",
            source_type="code",
            metadata={"entity_type": "function"},
        )
        results.append(SearchResult(document=doc2, score=0.85, rank=2))

        # Another document
        doc3 = Document(
            id="doc2",
            content="Learn about user management",
            title="User Management",
            source="confluence://docs/users",
            source_type="document",
            metadata={"space": "docs"},
        )
        results.append(SearchResult(document=doc3, score=0.75, rank=3))

        return results

    def test_retrieve_documents_by_query(
        self, retriever, mock_vector_store, mock_embedder, sample_search_results
    ):
        """Test basic retrieval by query."""
        # Setup mocks
        mock_embedder.embed.return_value = [0.1] * 384
        mock_vector_store.search.return_value = sample_search_results

        # Execute
        results = retriever.retrieve("authentication", limit=10)

        # Verify
        assert len(results) == 3
        assert results[0].document.title == "Authentication Guide"
        assert results[0].score == 0.95
        mock_embedder.embed.assert_called_once_with("authentication")
        mock_vector_store.search.assert_called_once()

    def test_retrieve_documents_filtered_by_source_type(
        self, retriever, mock_vector_store, mock_embedder, sample_search_results
    ):
        """Test retrieval with source_type filtering."""
        # Setup mocks
        mock_embedder.embed.return_value = [0.1] * 384
        doc_only = [r for r in sample_search_results if r.document.source_type == "document"]
        mock_vector_store.search.return_value = doc_only

        # Execute
        results = retriever.retrieve("authentication", limit=10, source_type="document")

        # Verify
        assert len(results) == 2
        assert all(r.document.source_type == "document" for r in results)
        # Check that filters were passed to vector_store.search
        call_args = mock_vector_store.search.call_args
        assert call_args[1].get("filters") == {"source_type": "document"}

    def test_retrieve_code_entities_by_query(
        self, retriever, mock_vector_store, mock_embedder, sample_search_results
    ):
        """Test code-specific retrieval."""
        # Setup mocks
        mock_embedder.embed.return_value = [0.1] * 384
        code_only = [r for r in sample_search_results if r.document.source_type == "code"]
        mock_vector_store.search.return_value = code_only

        # Execute
        results = retriever.retrieve_code("authenticate", limit=5)

        # Verify
        assert len(results) == 1
        assert results[0].document.source_type == "code"
        assert results[0].document.title == "authenticate"

    def test_retrieve_documents_only_by_query(
        self, retriever, mock_vector_store, mock_embedder, sample_search_results
    ):
        """Test document-specific retrieval."""
        # Setup mocks
        mock_embedder.embed.return_value = [0.1] * 384
        doc_only = [r for r in sample_search_results if r.document.source_type == "document"]
        mock_vector_store.search.return_value = doc_only

        # Execute
        results = retriever.retrieve_documents("authentication", limit=5)

        # Verify
        assert len(results) == 2
        assert all(r.document.source_type == "document" for r in results)

    def test_retrieve_hybrid_returns_both_categories(
        self, retriever, mock_vector_store, mock_embedder, sample_search_results
    ):
        """Test hybrid retrieval returns categorized results."""
        # Setup mocks
        mock_embedder.embed.return_value = [0.1] * 384
        mock_vector_store.search.side_effect = [
            [r for r in sample_search_results if r.document.source_type == "document"],
            [r for r in sample_search_results if r.document.source_type == "code"],
        ]

        # Execute
        results = retriever.retrieve_hybrid("authentication", limit=5)

        # Verify
        assert "documents" in results
        assert "code" in results
        assert len(results["documents"]) == 2
        assert len(results["code"]) == 1
        assert all(r.document.source_type == "document" for r in results["documents"])
        assert all(r.document.source_type == "code" for r in results["code"])

    def test_retrieve_with_custom_limit(
        self, retriever, mock_vector_store, mock_embedder, sample_search_results
    ):
        """Test custom limit parameter."""
        # Setup mocks
        mock_embedder.embed.return_value = [0.1] * 384
        mock_vector_store.search.return_value = sample_search_results[:1]

        # Execute
        results = retriever.retrieve("authentication", limit=1)

        # Verify
        call_args = mock_vector_store.search.call_args
        assert call_args[1].get("limit") == 1

    def test_retrieve_empty_results(self, retriever, mock_vector_store, mock_embedder):
        """Test handling of empty results."""
        # Setup mocks
        mock_embedder.embed.return_value = [0.1] * 384
        mock_vector_store.search.return_value = []

        # Execute
        results = retriever.retrieve("nonexistent query", limit=10)

        # Verify
        assert results == []
        assert len(results) == 0

    def test_retrieve_uses_default_limit(
        self, retriever, mock_vector_store, mock_embedder, sample_search_results
    ):
        """Test that default limit is used when not specified."""
        # Setup mocks
        mock_embedder.embed.return_value = [0.1] * 384
        mock_vector_store.search.return_value = sample_search_results

        # Execute
        results = retriever.retrieve("authentication")

        # Verify
        call_args = mock_vector_store.search.call_args
        assert call_args[1].get("limit") == 10

    def test_retrieve_with_none_limit_uses_default(
        self, retriever, mock_vector_store, mock_embedder, sample_search_results
    ):
        """Test that None limit uses default."""
        # Setup mocks
        mock_embedder.embed.return_value = [0.1] * 384
        mock_vector_store.search.return_value = sample_search_results

        # Execute
        results = retriever.retrieve("authentication", limit=None)

        # Verify
        call_args = mock_vector_store.search.call_args
        assert call_args[1].get("limit") == 10
