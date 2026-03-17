"""Document data model for ingestion system."""
from dataclasses import dataclass, field
from hashlib import md5
from typing import Dict, Any


@dataclass
class Document:
    """Represents a document that has been ingested from a source."""
    
    id: str                    # Unique identifier
    content: str              # Text content
    title: str                # Document title
    source: str               # Source identifier (file path, URL, etc)
    source_type: str          # Type of source: "file", "confluence", "code"
    metadata: Dict[str, Any] = field(default_factory=dict)  # Extra metadata
    
    def __post_init__(self):
        """Ensure ID is set."""
        if not self.id:
            # Generate deterministic ID from source and title
            combined = f"{self.source}:{self.title}"
            self.id = md5(combined.encode()).hexdigest()
