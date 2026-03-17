"""Vector store implementations for similarity search."""

from abc import ABC, abstractmethod
import json
import weaviate
from src.models.document import Document
from src.models.search_result import SearchResult


class VectorStore(ABC):
    """Abstract base class for vector store implementations."""

    @abstractmethod
    def add_documents(self, documents: list[Document], embeddings: list[list[float]]) -> None:
        """Add documents with pre-computed embeddings to the store.

        Args:
            documents: List of Document objects to add.
            embeddings: List of embedding vectors corresponding to documents.
        """
        pass

    @abstractmethod
    def search(
        self, query_embedding: list[float], limit: int = 10, filters: dict = None
    ) -> list[SearchResult]:
        """Search for similar documents using a query embedding.

        Args:
            query_embedding: Query vector for similarity search.
            limit: Maximum number of results to return.
            filters: Optional filters to apply to search.

        Returns:
            List of SearchResult objects sorted by score (descending).
        """
        pass

    @abstractmethod
    def delete_collection(self) -> None:
        """Delete all documents in this collection."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Count total documents stored in the collection.

        Returns:
            Number of documents in the collection.
        """
        pass


class WeaviateVectorStore(VectorStore):
    """Production vector store using Weaviate."""

    def __init__(
        self,
        host: str,
        port: int,
        collection_name: str,
        embedding_dimension: int,
        api_key: str = None,
    ):
        """Connect to Weaviate instance.

        Args:
            host: Hostname or URL of Weaviate instance.
            port: Port number for Weaviate (or None for cloud).
            collection_name: Name of the collection to use.
            embedding_dimension: Dimension of the embedding vectors.
            api_key: Optional API key for Weaviate Cloud.
        """
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        self.client = self._connect(host, port, api_key)
        self._ensure_collection_exists()

    def _connect(self, host: str, port: int, api_key: str = None):
        """Connect to Weaviate instance.

        Args:
            host: Hostname or URL.
            port: Port number.
            api_key: Optional API key for cloud.

        Returns:
            Weaviate client instance.
        """
        if api_key:
            # Cloud connection
            auth = weaviate.auth.AuthApiKey(api_key)
            client = weaviate.connect_to_weaviate_cloud(cluster_url=host, auth_credentials=auth)
        else:
            # Local connection
            client = weaviate.connect_to_local(host=host, port=port)

        return client

    def _ensure_collection_exists(self):
        """Create collection if it doesn't exist."""
        try:
            # Try to get existing collection
            self.client.collections.get(self.collection_name)
        except Exception:
            # Collection doesn't exist, create it
            from weaviate.collections.classes.config import Configure

            self.client.collections.create(
                name=self.collection_name,
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Configure.Property(name="id", data_type=Configure.DataType.TEXT),
                    Configure.Property(name="content", data_type=Configure.DataType.TEXT),
                    Configure.Property(name="title", data_type=Configure.DataType.TEXT),
                    Configure.Property(name="source", data_type=Configure.DataType.TEXT),
                    Configure.Property(name="source_type", data_type=Configure.DataType.TEXT),
                    Configure.Property(name="metadata", data_type=Configure.DataType.TEXT),
                ],
            )

    def add_documents(self, documents: list[Document], embeddings: list[list[float]]) -> None:
        """Add documents with pre-computed embeddings to the store.

        Args:
            documents: List of Document objects.
            embeddings: List of embedding vectors.
        """
        collection = self.client.collections.get(self.collection_name)

        # Prepare objects to insert
        objects_to_insert = []
        for doc, embedding in zip(documents, embeddings):
            obj = {
                "id": doc.id,
                "content": doc.content,
                "title": doc.title,
                "source": doc.source,
                "source_type": doc.source_type,
                "metadata": json.dumps(doc.metadata),
            }
            objects_to_insert.append(obj)

        # Add to collection
        collection.data.insert_multiple(objects=objects_to_insert, vectors=embeddings)

    def search(
        self, query_embedding: list[float], limit: int = 10, filters: dict = None
    ) -> list[SearchResult]:
        """Search for similar documents using a query embedding.

        Args:
            query_embedding: Query embedding vector.
            limit: Maximum number of results.
            filters: Optional filters (not yet implemented).

        Returns:
            List of SearchResult objects sorted by score.
        """
        collection = self.client.collections.get(self.collection_name)

        # Perform near vector search
        query_builder = collection.query.near_vector(near_vector=query_embedding)
        response = query_builder.with_limit(limit)

        # Get objects from response
        objects = response.objects if hasattr(response, "objects") else []

        # Convert results to SearchResult objects
        results = []
        for rank, obj in enumerate(objects, start=1):
            # Extract document fields
            props = obj.properties if hasattr(obj, "properties") else {}
            if isinstance(props, dict):
                doc_data = props
            else:
                doc_data = {
                    "id": getattr(obj, "id", ""),
                    "content": getattr(obj, "content", ""),
                    "title": getattr(obj, "title", ""),
                    "source": getattr(obj, "source", ""),
                    "source_type": getattr(obj, "source_type", ""),
                    "metadata": getattr(obj, "metadata", "{}"),
                }

            doc = Document(
                id=doc_data.get("id", ""),
                content=doc_data.get("content", ""),
                title=doc_data.get("title", ""),
                source=doc_data.get("source", ""),
                source_type=doc_data.get("source_type", ""),
                metadata=json.loads(doc_data.get("metadata", "{}")),
            )

            # Get similarity score
            metadata = getattr(obj, "metadata", None)
            score = getattr(metadata, "certainty", 0.0) if metadata else 0.0

            result = SearchResult(document=doc, score=score, rank=rank)
            results.append(result)

        # Sort by score descending (should already be sorted, but ensure it)
        results.sort(key=lambda r: r.score, reverse=True)

        # Update ranks
        for i, result in enumerate(results, start=1):
            result.rank = i

        return results

    def delete_collection(self) -> None:
        """Delete the collection."""
        self.client.collections.delete(self.collection_name)

    def count(self) -> int:
        """Count documents in the collection.

        Returns:
            Number of documents.
        """
        collection = self.client.collections.get(self.collection_name)
        response = collection.aggregate.over_all().total_count()
        total = getattr(response, "total_count", 0)
        return total if isinstance(total, int) else 0
