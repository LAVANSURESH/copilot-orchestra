"""SearchResult data model for vector search results."""

from dataclasses import dataclass
from src.models.document import Document


@dataclass
class SearchResult:
    """Represents a document search result with similarity score."""

    document: Document  # The matching document
    score: float  # Similarity score (0-1)
    rank: int  # Position in results (1-indexed)
