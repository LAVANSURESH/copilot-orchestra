"""SemanticMapper for establishing relationships between code and documentation."""

from src.retrieval.retriever import SemanticRetriever
from src.models.document import Document
from src.models.code_entity import CodeEntity
from src.models.search_result import SearchResult


class SemanticMapper:
    """Establishes semantic relationships between code entities and documentation."""

    def __init__(self, retriever: SemanticRetriever, similarity_threshold: float = 0.5):
        """Initialize the semantic mapper.

        Args:
            retriever: SemanticRetriever to search for related content.
            similarity_threshold: Minimum similarity score to consider a result related (0-1).
        """
        self.retriever = retriever
        self.similarity_threshold = similarity_threshold

    def find_related_docs_for_code(
        self, code_entity: CodeEntity, limit: int = 5
    ) -> list[SearchResult]:
        """Find documentation related to a code entity.

        Args:
            code_entity: CodeEntity to find related documentation for.
            limit: Maximum number of results to return.

        Returns:
            List of SearchResult objects for related documentation.
        """
        # Build a rich query from entity name, docstring, and parameters
        query_parts = [code_entity.name]

        if code_entity.docstring:
            query_parts.append(code_entity.docstring)

        if code_entity.parameters:
            query_parts.append(" ".join(code_entity.parameters))

        query = " ".join(query_parts)

        # Retrieve documents
        results = self.retriever.retrieve_documents(query, limit=limit)

        # Filter by similarity threshold
        filtered_results = [r for r in results if r.score >= self.similarity_threshold]

        return filtered_results

    def find_related_code_for_doc(self, document: Document, limit: int = 5) -> list[SearchResult]:
        """Find code entities related to a document.

        Args:
            document: Document to find related code for.
            limit: Maximum number of results to return.

        Returns:
            List of SearchResult objects for related code entities.
        """
        # Build query from document title and content
        query_parts = [document.title]

        # Add first 100 words of content for context
        content_words = document.content.split()[:50]
        if content_words:
            query_parts.append(" ".join(content_words))

        query = " ".join(query_parts)

        # Retrieve code
        results = self.retriever.retrieve_code(query, limit=limit)

        # Filter by similarity threshold
        filtered_results = [r for r in results if r.score >= self.similarity_threshold]

        return filtered_results

    def build_mapping(
        self, code_entities: list[CodeEntity], documents: list[Document]
    ) -> dict[str, list[str]]:
        """Build bidirectional mapping between code and documentation.

        Args:
            code_entities: List of CodeEntity objects.
            documents: List of Document objects.

        Returns:
            Dictionary with bidirectional mappings:
            {"code:{entity_name}": ["doc:{doc_id}", ...], "doc:{doc_id}": ["code:{entity_name}", ...]}
        """
        mappings = {}

        # Build maps for forward and reverse lookups
        for entity in code_entities:
            entity_key = f"code:{entity.name}"
            related_docs = self.find_related_docs_for_code(entity)

            if related_docs:
                mappings[entity_key] = [f"doc:{r.document.id}" for r in related_docs]

        # Build reverse mappings
        for document in documents:
            doc_key = f"doc:{document.id}"
            related_code = self.find_related_code_for_doc(document)

            if related_code:
                code_ids = [f"code:{r.document.title}" for r in related_code]

                # Combine with existing mappings if any
                if doc_key in mappings:
                    # Avoid duplicates
                    existing = set(mappings[doc_key])
                    mappings[doc_key] = list(existing | set(code_ids))
                else:
                    mappings[doc_key] = code_ids

        return mappings
