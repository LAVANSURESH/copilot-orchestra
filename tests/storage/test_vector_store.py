"""Tests for VectorStore implementations."""
import pytest
from unittest.mock import patch, MagicMock, call
from src.storage.vector_store import VectorStore, WeaviateVectorStore
from src.models.document import Document
from src.models.search_result import SearchResult


class TestVectorStoreInterface:
    """Test the abstract VectorStore interface."""
    
    def test_vector_store_cannot_be_instantiated(self):
        """Test that VectorStore abstract class cannot be instantiated."""
        with pytest.raises(TypeError):
            VectorStore()


class TestWeaviateVectorStore:
    """Test cases for WeaviateVectorStore implementation."""
    
    @patch('src.storage.vector_store.weaviate')
    def test_weaviate_connection_local(self, mock_weaviate):
        """Test connecting to local Weaviate instance."""
        mock_client = MagicMock()
        mock_weaviate.connect_to_local.return_value = mock_client
        
        store = WeaviateVectorStore(
            host="localhost",
            port=8080,
            collection_name="TestCollection",
            embedding_dimension=384
        )
        
        mock_weaviate.connect_to_local.assert_called_once_with(
            host="localhost",
            port=8080
        )
        assert store.client is not None
    
    @patch('src.storage.vector_store.weaviate')
    def test_add_documents_to_store(self, mock_weaviate):
        """Test adding documents with embeddings to the store."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.collections.get.return_value = mock_collection
        mock_weaviate.connect_to_local.return_value = mock_client
        
        store = WeaviateVectorStore(
            host="localhost",
            port=8080,
            collection_name="TestCollection",
            embedding_dimension=384
        )
        
        # Create test documents
        doc1 = Document(
            id="doc1",
            content="Test content 1",
            title="Document 1",
            source="test1.md",
            source_type="file"
        )
        doc2 = Document(
            id="doc2",
            content="Test content 2",
            title="Document 2",
            source="test2.md",
            source_type="file"
        )
        
        embeddings = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ]
        
        store.add_documents([doc1, doc2], embeddings)
        
        # Verify add_objects was called
        assert mock_collection.data.insert.called or mock_collection.data.insert_multiple.called
    
    @patch('src.storage.vector_store.weaviate')
    def test_similarity_search(self, mock_weaviate):
        """Test vector similarity search."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        
        # Mock search result
        mock_object = MagicMock()
        mock_object.properties = {
            'id': 'doc1',
            'content': 'Test content',
            'title': 'Test Document',
            'source': 'test.md',
            'source_type': 'file',
            'metadata': '{}'
        }
        mock_object.metadata.certainty = 0.95
        
        # Set up the chained mock calls
        mock_query_response = MagicMock()
        mock_query_response.objects = [mock_object]
        
        mock_near_vector_response = MagicMock()
        mock_near_vector_response.with_limit.return_value = mock_query_response
        
        mock_collection.query.near_vector.return_value = mock_near_vector_response
        mock_client.collections.get.return_value = mock_collection
        mock_weaviate.connect_to_local.return_value = mock_client
        
        store = WeaviateVectorStore(
            host="localhost",
            port=8080,
            collection_name="TestCollection",
            embedding_dimension=384
        )
        
        query_embedding = [0.1, 0.2, 0.3]
        results = store.search(query_embedding, limit=10)
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert isinstance(results[0], SearchResult)
    
    @patch('src.storage.vector_store.weaviate')
    def test_search_with_limit(self, mock_weaviate):
        """Test that search respects limit parameter."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        
        # Create mock results
        mock_objects = []
        for i in range(5):
            obj = MagicMock()
            obj.properties = {
                'id': f'doc{i}',
                'content': f'Content {i}',
                'title': f'Title {i}',
                'source': f'source{i}.md',
                'source_type': 'file',
                'metadata': '{}'
            }
            obj.metadata.certainty = 0.9 - (i * 0.1)
            mock_objects.append(obj)
        
        # Set up the chained mock calls
        mock_query_response = MagicMock()
        mock_query_response.objects = mock_objects[:3]
        
        mock_near_vector_response = MagicMock()
        mock_near_vector_response.with_limit.return_value = mock_query_response
        
        mock_collection.query.near_vector.return_value = mock_near_vector_response
        mock_client.collections.get.return_value = mock_collection
        mock_weaviate.connect_to_local.return_value = mock_client
        
        store = WeaviateVectorStore(
            host="localhost",
            port=8080,
            collection_name="TestCollection",
            embedding_dimension=384
        )
        
        results = store.search([0.1, 0.2], limit=3)
        
        assert len(results) == 3
    
    @patch('src.storage.vector_store.weaviate')
    def test_delete_collection(self, mock_weaviate):
        """Test deleting a collection."""
        mock_client = MagicMock()
        mock_weaviate.connect_to_local.return_value = mock_client
        
        store = WeaviateVectorStore(
            host="localhost",
            port=8080,
            collection_name="TestCollection",
            embedding_dimension=384
        )
        
        store.delete_collection()
        
        mock_client.collections.delete.assert_called_once_with("TestCollection")
    
    @patch('src.storage.vector_store.weaviate')
    def test_count_documents(self, mock_weaviate):
        """Test counting documents in collection."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        
        # Mock the aggregation result with proper chaining
        mock_count_result = MagicMock()
        mock_count_result.total_count = 42
        
        mock_over_result = MagicMock()
        mock_over_result.total_count.return_value = mock_count_result
        
        mock_collection.aggregate.over_all.return_value = mock_over_result
        mock_client.collections.get.return_value = mock_collection
        mock_weaviate.connect_to_local.return_value = mock_client
        
        store = WeaviateVectorStore(
            host="localhost",
            port=8080,
            collection_name="TestCollection",
            embedding_dimension=384
        )
        
        count = store.count()
        
        assert isinstance(count, int)
        assert count == 42
    
    @patch('src.storage.vector_store.weaviate')
    def test_search_returns_sorted_by_score(self, mock_weaviate):
        """Test that search results are sorted by score (descending)."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        
        # Create mock results with different scores
        mock_objects = []
        scores = [0.95, 0.87, 0.76]  # Already sorted descending
        for i, score in enumerate(scores):
            obj = MagicMock()
            obj.properties = {
                'id': f'doc{i}',
                'content': f'Content {i}',
                'title': f'Title {i}',
                'source': f'source{i}.md',
                'source_type': 'file',
                'metadata': '{}'
            }
            obj.metadata.certainty = score
            mock_objects.append(obj)
        
        # Set up the chained mock calls
        mock_query_response = MagicMock()
        mock_query_response.objects = mock_objects
        
        mock_near_vector_response = MagicMock()
        mock_near_vector_response.with_limit.return_value = mock_query_response
        
        mock_collection.query.near_vector.return_value = mock_near_vector_response
        mock_client.collections.get.return_value = mock_collection
        mock_weaviate.connect_to_local.return_value = mock_client
        
        store = WeaviateVectorStore(
            host="localhost",
            port=8080,
            collection_name="TestCollection",
            embedding_dimension=384
        )
        
        results = store.search([0.1, 0.2], limit=10)
        
        # Check that results are sorted by score
        assert len(results) == 3
        assert results[0].rank == 1
        assert results[1].rank == 2
        assert results[2].rank == 3
        
        # Verify scores are in descending order
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score
    
    @patch('src.storage.vector_store.weaviate')
    def test_weaviate_cloud_connection(self, mock_weaviate):
        """Test connecting to Weaviate Cloud."""
        mock_client = MagicMock()
        mock_auth = MagicMock()
        mock_weaviate.auth.AuthApiKey.return_value = mock_auth
        mock_weaviate.connect_to_weaviate_cloud.return_value = mock_client
        
        store = WeaviateVectorStore(
            host="https://cluster-id.weaviate.cloud",
            port=443,
            collection_name="TestCollection",
            embedding_dimension=384,
            api_key="test-api-key"
        )
        
        # Should have created auth
        mock_weaviate.auth.AuthApiKey.assert_called_once_with("test-api-key")
        
        # Should have connected to cloud
        mock_weaviate.connect_to_weaviate_cloud.assert_called_once()
    
    @patch('src.storage.vector_store.weaviate')
    def test_search_with_filters(self, mock_weaviate):
        """Test search with optional filters."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        
        mock_object = MagicMock()
        mock_object.properties = {
            'id': 'doc1',
            'content': 'Test',
            'title': 'Test',
            'source': 'test.md',
            'source_type': 'file',
            'metadata': '{}'
        }
        mock_object.metadata.certainty = 0.95
        
        # Set up the chained mock calls
        mock_query_response = MagicMock()
        mock_query_response.objects = [mock_object]
        
        mock_near_vector_response = MagicMock()
        mock_near_vector_response.with_limit.return_value = mock_query_response
        
        mock_collection.query.near_vector.return_value = mock_near_vector_response
        mock_client.collections.get.return_value = mock_collection
        mock_weaviate.connect_to_local.return_value = mock_client
        
        store = WeaviateVectorStore(
            host="localhost",
            port=8080,
            collection_name="TestCollection",
            embedding_dimension=384
        )
        
        # Should accept filters parameter
        results = store.search([0.1, 0.2], limit=10, filters={"source_type": "file"})
        
        assert isinstance(results, list)
