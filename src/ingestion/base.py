"""Abstract base class for document sources."""
from abc import ABC, abstractmethod
from typing import List
from src.models.document import Document


class DocumentSource(ABC):
    """Abstract base class for document sources."""

    @abstractmethod
    def load(self) -> List[Document]:
        """Load documents from source.

        Returns:
            List of Document objects
        """
        pass

    @abstractmethod
    def get_source_id(self) -> str:
        """Get unique identifier for this source.

        Returns:
            String identifier for the source
        """
        pass
