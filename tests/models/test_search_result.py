"""Tests for SearchResult model."""
import pytest
from src.models.search_result import SearchResult
from src.models.document import Document


class TestSearchResult:
    """Test cases for SearchResult dataclass."""
    
    def test_search_result_creation(self):
        """Test creating a SearchResult instance."""
        doc = Document(
            id="test-id",
            content="Test content",
            title="Test Document",
            source="test.md",
            source_type="file"
        )
        
        result = SearchResult(
            document=doc,
            score=0.95,
            rank=1
        )
        
        assert result.document == doc
        assert result.score == 0.95
        assert result.rank == 1
    
    def test_search_result_score_range(self):
        """Test that score is between 0 and 1."""
        doc = Document(
            id="test-id",
            content="Test content",
            title="Test Document",
            source="test.md",
            source_type="file"
        )
        
        # Valid scores
        result_high = SearchResult(document=doc, score=1.0, rank=1)
        result_low = SearchResult(document=doc, score=0.0, rank=1)
        result_mid = SearchResult(document=doc, score=0.5, rank=1)
        
        assert result_high.score == 1.0
        assert result_low.score == 0.0
        assert result_mid.score == 0.5
    
    def test_search_result_rank_positive(self):
        """Test that rank is a positive integer."""
        doc = Document(
            id="test-id",
            content="Test content",
            title="Test Document",
            source="test.md",
            source_type="file"
        )
        
        result = SearchResult(document=doc, score=0.95, rank=5)
        assert result.rank == 5
        assert isinstance(result.rank, int)
