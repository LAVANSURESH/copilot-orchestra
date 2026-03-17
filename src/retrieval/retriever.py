"""SemanticRetriever for semantic search queries."""

from src.storage.vector_store import VectorStore
from src.embeddings.embedder import EmbeddingModel
from src.models.search_result import SearchResult


class SemanticRetriever:
    """Retrieves relevant documents using semantic similarity search."""

    def __init__(
        self, vector_store: VectorStore, embedder: EmbeddingModel, default_limit: int = 10
    ):
        """Initialize the retriever.

        Args:
            vector_store: Vector store to search in.
            embedder: Embedding model to encode queries.
            default_limit: Default number of results to return.
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.default_limit = default_limit

    def retrieve(
        self, query: str, limit: int = None, source_type: str = None
    ) -> list[SearchResult]:
        """Retrieve relevant documents by semantic similarity.

        Args:
            query: Natural language query string.
            limit: Maximum number of results to return. Uses default_limit if None.
            source_type: Optional filter by source type (e.g., "document", "code").

        Returns:
            List of SearchResult objects sorted by score (descending).
        """
        # Use default limit if not specified
        if limit is None:
            limit = self.default_limit

        # Embed the query
        query_embedding = self.embedder.embed(query)

        # Build filters if source_type is specified
        filters = None
        if source_type is not None:
            filters = {"source_type": source_type}

        # Search the vector store
        results = self.vector_store.search(query_embedding, limit=limit, filters=filters)

        return results

    def retrieve_documents(self, query: str, limit: int = None) -> list[SearchResult]:
        """Retrieve only document-type results.

        Args:
            query: Natural language query string.
            limit: Maximum number of results to return.

        Returns:
            List of SearchResult objects for documents only.
        """
        return self.retrieve(query, limit=limit, source_type="document")

    def retrieve_code(self, query: str, limit: int = None) -> list[SearchResult]:
        """Retrieve only code-type results.

        Args:
            query: Natural language query string.
            limit: Maximum number of results to return.

        Returns:
            List of SearchResult objects for code entities only.
        """
        return self.retrieve(query, limit=limit, source_type="code")

    def retrieve_hybrid(self, query: str, limit: int = None) -> dict[str, list[SearchResult]]:
        """Retrieve both documents and code, returning categorized results.

        Args:
            query: Natural language query string.
            limit: Maximum number of results per category.

        Returns:
            Dictionary with "documents" and "code" keys, each containing SearchResult lists.
        """
        return {
            "documents": self.retrieve_documents(query, limit),
            "code": self.retrieve_code(query, limit),
        }
