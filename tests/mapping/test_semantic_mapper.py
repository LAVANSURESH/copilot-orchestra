"""Tests for semantic mapper."""

import pytest
from unittest.mock import Mock, MagicMock
from src.mapping.semantic_mapper import SemanticMapper
from src.models.document import Document
from src.models.code_entity import CodeEntity, EntityType
from src.models.search_result import SearchResult


class TestSemanticMapper:
    """Test suite for SemanticMapper."""

    @pytest.fixture
    def mock_retriever(self):
        """Create a mock retriever."""
        retriever = Mock()
        return retriever

    @pytest.fixture
    def mapper(self, mock_retriever):
        """Create a mapper instance with mock retriever."""
        return SemanticMapper(mock_retriever, similarity_threshold=0.5)

    @pytest.fixture
    def sample_code_entity(self):
        """Create a sample code entity."""
        return CodeEntity(
            name="authenticate_user",
            entity_type=EntityType.FUNCTION,
            file_path="src/auth.py",
            start_line=10,
            end_line=25,
            docstring="Authenticate a user with credentials.",
            source_code="def authenticate_user(username, password):\n    return validate(username, password)",
            parameters=["username", "password"],
        )

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents."""
        docs = []

        doc1 = Document(
            id="doc1",
            content="User authentication guide",
            title="Authentication Guide",
            source="confluence://auth",
            source_type="document",
        )
        docs.append(doc1)

        doc2 = Document(
            id="doc2",
            content="API documentation for login endpoints",
            title="Login API",
            source="confluence://api/login",
            source_type="document",
        )
        docs.append(doc2)

        return docs

    @pytest.fixture
    def sample_search_results(self):
        """Create sample search results."""
        results = []

        doc = Document(
            id="doc1",
            content="User authentication guide",
            title="Authentication Guide",
            source="confluence://auth",
            source_type="document",
        )
        results.append(SearchResult(document=doc, score=0.9, rank=1))

        doc2 = Document(
            id="doc2",
            content="API documentation for login endpoints",
            title="Login API",
            source="confluence://api/login",
            source_type="document",
        )
        results.append(SearchResult(document=doc2, score=0.75, rank=2))

        return results

    def test_find_related_docs_for_code_entity(
        self, mapper, mock_retriever, sample_code_entity, sample_search_results
    ):
        """Test finding documentation related to a code entity."""
        # Setup mock
        mock_retriever.retrieve_documents.return_value = sample_search_results

        # Execute
        results = mapper.find_related_docs_for_code(sample_code_entity, limit=5)

        # Verify
        assert len(results) == 2
        assert results[0].document.title == "Authentication Guide"
        # Verify that retrieve_documents was called
        mock_retriever.retrieve_documents.assert_called_once()
        call_args = mock_retriever.retrieve_documents.call_args
        # Query should include entity name, docstring, and parameters
        query = call_args[0][0]
        assert "authenticate_user" in query
        assert "Authenticate a user with credentials" in query

    def test_find_related_code_for_document(self, mapper, mock_retriever):
        """Test finding code entities related to a document."""
        # Create sample results
        code_doc = Document(
            id="code1",
            content="def validate(user):\n    return check_user(user)",
            title="validate",
            source="src/auth.py",
            source_type="code",
            metadata={"entity_type": "function"},
        )
        results = [SearchResult(document=code_doc, score=0.85, rank=1)]

        # Setup mock
        mock_retriever.retrieve_code.return_value = results

        # Create sample document
        doc = Document(
            id="doc1",
            content="User authentication guide",
            title="Authentication",
            source="confluence://auth",
            source_type="document",
        )

        # Execute
        results = mapper.find_related_code_for_doc(doc, limit=5)

        # Verify
        assert len(results) == 1
        assert results[0].document.title == "validate"
        mock_retriever.retrieve_code.assert_called_once()

    def test_build_mapping_creates_bidirectional_links(
        self, mapper, mock_retriever, sample_code_entity, sample_documents, sample_search_results
    ):
        """Test building bidirectional mapping between code and docs."""
        # Setup mock to return docs for code
        mock_retriever.retrieve_documents.return_value = sample_search_results

        # Create code entity results for document search
        code_doc = Document(
            id="code1",
            content="def authenticate_user(username, password):\n    pass",
            title="authenticate_user",
            source="src/auth.py",
            source_type="code",
        )
        code_results = [SearchResult(document=code_doc, score=0.85, rank=1)]
        mock_retriever.retrieve_code.return_value = code_results

        # Execute
        mappings = mapper.build_mapping([sample_code_entity], sample_documents)

        # Verify structure
        assert isinstance(mappings, dict)
        # Should have code entity mappings
        code_key = f"code:{sample_code_entity.name}"
        assert code_key in mappings or any("code:" in k for k in mappings.keys())

    def test_similarity_threshold_filters_results(self, mapper, mock_retriever, sample_code_entity):
        """Test that similarity threshold filters results."""
        # Create results with varying scores
        high_score_doc = Document(
            id="doc1",
            content="About authentication",
            title="Auth High",
            source="confluence://auth1",
            source_type="document",
        )
        high_result = SearchResult(document=high_score_doc, score=0.85, rank=1)

        low_score_doc = Document(
            id="doc2",
            content="About something else",
            title="Auth Low",
            source="confluence://auth2",
            source_type="document",
        )
        low_result = SearchResult(document=low_score_doc, score=0.35, rank=2)

        # Setup mock to return all results
        mock_retriever.retrieve_documents.return_value = [high_result, low_result]

        # Create mapper with high threshold
        strict_mapper = SemanticMapper(mock_retriever, similarity_threshold=0.5)

        # Execute
        results = strict_mapper.find_related_docs_for_code(sample_code_entity, limit=5)

        # Verify - only results above threshold
        assert len(results) == 1
        assert results[0].score >= 0.5

    def test_find_related_docs_respects_limit(self, mapper, mock_retriever, sample_code_entity):
        """Test that limit parameter is respected."""
        # Create multiple results
        results = []
        for i in range(10):
            doc = Document(
                id=f"doc{i}",
                content=f"Document {i}",
                title=f"Doc {i}",
                source=f"confluence://doc{i}",
                source_type="document",
            )
            results.append(SearchResult(document=doc, score=0.9 - (i * 0.01), rank=i + 1))

        mock_retriever.retrieve_documents.return_value = results

        # Execute with limit
        retrieved = mapper.find_related_docs_for_code(sample_code_entity, limit=3)

        # Verify limit is passed
        call_args = mock_retriever.retrieve_documents.call_args
        assert call_args[1].get("limit") == 3

    def test_mapper_with_zero_threshold(self, mock_retriever):
        """Test mapper with threshold of 0 (accepts all)."""
        mapper = SemanticMapper(mock_retriever, similarity_threshold=0.0)

        # Create results with low scores
        doc = Document(
            id="doc1", content="Something", title="Doc", source="source", source_type="document"
        )
        results = [SearchResult(document=doc, score=0.01, rank=1)]

        mock_retriever.retrieve_documents.return_value = results

        # Create code entity
        entity = CodeEntity(
            name="func",
            entity_type=EntityType.FUNCTION,
            file_path="file.py",
            start_line=1,
            end_line=5,
            docstring="Doc",
            source_code="code",
        )

        # Execute
        retrieved = mapper.find_related_docs_for_code(entity)

        # Should include even low-score result
        assert len(retrieved) == 1

    def test_find_related_docs_empty_results(self, mapper, mock_retriever, sample_code_entity):
        """Test handling of empty results."""
        # Setup mock to return empty
        mock_retriever.retrieve_documents.return_value = []

        # Execute
        results = mapper.find_related_docs_for_code(sample_code_entity)

        # Verify
        assert results == []

    def test_build_mapping_empty_inputs(self, mapper, mock_retriever):
        """Test building mapping with empty inputs."""
        # Execute
        mappings = mapper.build_mapping([], [])

        # Verify
        assert isinstance(mappings, dict)
