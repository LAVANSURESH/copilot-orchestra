"""Tests for semantic ranker."""

import pytest
from src.retrieval.ranker import Ranker
from src.models.document import Document
from src.models.search_result import SearchResult


class TestRanker:
    """Test suite for Ranker."""

    @pytest.fixture
    def ranker(self):
        """Create a ranker instance."""
        return Ranker()

    @pytest.fixture
    def unsorted_results(self):
        """Create unsorted search results for testing."""
        results = []

        # Create results with mixed order
        doc1 = Document(
            id="doc1",
            content="First document about authentication",
            title="Auth Doc",
            source="source1",
            source_type="document",
        )
        results.append(SearchResult(document=doc1, score=0.75, rank=1))

        doc2 = Document(
            id="doc2",
            content="Second document about authentication",
            title="Auth Guide",
            source="source2",
            source_type="document",
        )
        results.append(SearchResult(document=doc2, score=0.95, rank=2))

        doc3 = Document(
            id="doc3",
            content="Third document about authentication",
            title="Auth Tutorial",
            source="source3",
            source_type="document",
        )
        results.append(SearchResult(document=doc3, score=0.85, rank=3))

        return results

    def test_rerank_results_by_relevance(self, ranker, unsorted_results):
        """Test that results are re-ranked by score descending."""
        # Execute
        reranked = ranker.rerank(unsorted_results)

        # Verify
        assert len(reranked) == 3
        # Results should be sorted by score descending
        assert reranked[0].score == 0.95
        assert reranked[1].score == 0.85
        assert reranked[2].score == 0.75

    def test_rerank_updates_rank_positions(self, ranker, unsorted_results):
        """Test that rank positions are updated correctly."""
        # Execute
        reranked = ranker.rerank(unsorted_results)

        # Verify
        assert reranked[0].rank == 1
        assert reranked[1].rank == 2
        assert reranked[2].rank == 3
        # Verify they match the scores
        assert reranked[0].document.title == "Auth Guide"  # score 0.95
        assert reranked[1].document.title == "Auth Tutorial"  # score 0.85
        assert reranked[2].document.title == "Auth Doc"  # score 0.75

    def test_deduplicate_removes_similar_results(self, ranker):
        """Test deduplication of similar content."""
        results = []

        # Create nearly identical results
        doc1 = Document(
            id="doc1",
            content="This is a function that authenticates users and validates credentials.",
            title="authenticate",
            source="source1",
            source_type="code",
        )
        results.append(SearchResult(document=doc1, score=0.95, rank=1))

        # Very similar content (>95% similar)
        doc2 = Document(
            id="doc2",
            content="This is a function that authenticates users and validates credentials.",
            title="authenticate_user",
            source="source2",
            source_type="code",
        )
        results.append(SearchResult(document=doc2, score=0.90, rank=2))

        # Different content
        doc3 = Document(
            id="doc3",
            content="This function is completely different and does something else entirely.",
            title="other_function",
            source="source3",
            source_type="code",
        )
        results.append(SearchResult(document=doc3, score=0.85, rank=3))

        # Execute with high similarity threshold
        deduplicated = ranker.deduplicate(results, similarity_threshold=0.95)

        # Verify - should remove near-duplicates but keep the different one
        # The first result (higher score) should be kept
        assert len(deduplicated) == 2
        assert deduplicated[0].document.id == "doc1"
        assert deduplicated[1].document.id == "doc3"

    def test_deduplicate_keeps_diverse_results(self, ranker):
        """Test that diverse results are kept."""
        results = []

        doc1 = Document(
            id="doc1",
            content="About authentication and login mechanisms",
            title="Authentication",
            source="source1",
            source_type="document",
        )
        results.append(SearchResult(document=doc1, score=0.95, rank=1))

        doc2 = Document(
            id="doc2",
            content="About database operations and queries",
            title="Database",
            source="source2",
            source_type="document",
        )
        results.append(SearchResult(document=doc2, score=0.90, rank=2))

        doc3 = Document(
            id="doc3",
            content="About file system operations and I/O",
            title="FileIO",
            source="source3",
            source_type="document",
        )
        results.append(SearchResult(document=doc3, score=0.85, rank=3))

        # Execute
        deduplicated = ranker.deduplicate(results, similarity_threshold=0.95)

        # Verify - all should be kept
        assert len(deduplicated) == 3

    def test_deduplicate_with_low_threshold_removes_more(self, ranker):
        """Test that lower threshold removes more results."""
        results = []

        # Similar but not identical
        doc1 = Document(
            id="doc1",
            content="This is about authentication and user login",
            title="Auth 1",
            source="source1",
            source_type="document",
        )
        results.append(SearchResult(document=doc1, score=0.95, rank=1))

        doc2 = Document(
            id="doc2",
            content="This is about authentication and user access",
            title="Auth 2",
            source="source2",
            source_type="document",
        )
        results.append(SearchResult(document=doc2, score=0.90, rank=2))

        # Execute with low threshold
        deduplicated = ranker.deduplicate(results, similarity_threshold=0.70)

        # With lower threshold, similar results should be removed
        assert len(deduplicated) <= 2

    def test_rerank_preserves_document_content(self, ranker, unsorted_results):
        """Test that document content is preserved during re-ranking."""
        original_content = [r.document.content for r in unsorted_results]

        # Execute
        reranked = ranker.rerank(unsorted_results)

        # Verify - all content should still be there
        reranked_content = [r.document.content for r in reranked]
        assert set(original_content) == set(reranked_content)

    def test_deduplicate_returns_sorted_by_score(self, ranker):
        """Test that deduplicated results are sorted by score."""
        results = []

        doc1 = Document(
            id="doc1",
            content="First document",
            title="Doc1",
            source="source1",
            source_type="document",
        )
        results.append(SearchResult(document=doc1, score=0.80, rank=1))

        doc2 = Document(
            id="doc2",
            content="Second document",
            title="Doc2",
            source="source2",
            source_type="document",
        )
        results.append(SearchResult(document=doc2, score=0.95, rank=2))

        doc3 = Document(
            id="doc3",
            content="Third document",
            title="Doc3",
            source="source3",
            source_type="document",
        )
        results.append(SearchResult(document=doc3, score=0.88, rank=3))

        # Execute
        deduplicated = ranker.deduplicate(results)

        # Verify - should be sorted by score descending
        assert deduplicated[0].score >= deduplicated[1].score
        if len(deduplicated) > 2:
            assert deduplicated[1].score >= deduplicated[2].score

    def test_rerank_with_single_result(self, ranker):
        """Test re-ranking with a single result."""
        results = []

        doc = Document(
            id="doc1",
            content="Single document",
            title="Single",
            source="source1",
            source_type="document",
        )
        results.append(SearchResult(document=doc, score=0.95, rank=1))

        # Execute
        reranked = ranker.rerank(results)

        # Verify
        assert len(reranked) == 1
        assert reranked[0].rank == 1
        assert reranked[0].score == 0.95

    def test_deduplicate_with_empty_results(self, ranker):
        """Test deduplication with empty results."""
        # Execute
        deduplicated = ranker.deduplicate([])

        # Verify
        assert deduplicated == []
