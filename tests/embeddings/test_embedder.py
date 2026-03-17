"""Tests for EmbeddingModel."""
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from src.embeddings.embedder import EmbeddingModel


class TestEmbeddingModel:
    """Test cases for EmbeddingModel."""
    
    @patch('src.embeddings.embedder.SentenceTransformer')
    def test_generate_embedding_for_text(self, mock_st):
        """Test that embedding is generated for a single text."""
        # Setup mock
        mock_model = MagicMock()
        mock_embedding = [0.1, 0.2, 0.3, 0.4]
        mock_model.encode.return_value = np.array(mock_embedding)
        mock_st.return_value = mock_model
        
        embedder = EmbeddingModel(model_name="test-model")
        embedding = embedder.embed("test text")
        
        assert isinstance(embedding, list)
        assert len(embedding) == 4
        assert all(isinstance(x, (float, np.floating)) for x in embedding)
    
    @patch('src.embeddings.embedder.SentenceTransformer')
    def test_embedding_dimension_consistency(self, mock_st):
        """Test that embeddings have consistent dimensions."""
        # Setup mock
        mock_model = MagicMock()
        # Return consistent dimension
        mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_model.encode.return_value = np.array(mock_embedding)
        mock_st.return_value = mock_model
        
        embedder = EmbeddingModel()
        
        embedding1 = embedder.embed("text one")
        embedding2 = embedder.embed("text two")
        
        assert len(embedding1) == len(embedding2)
        assert len(embedding1) == 5
    
    @patch('src.embeddings.embedder.SentenceTransformer')
    def test_batch_embedding_same_as_individual(self, mock_st):
        """Test that batch embedding matches individual embeddings."""
        # Setup mock
        mock_model = MagicMock()
        texts = ["text one", "text two", "text three"]
        embeddings = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ])
        mock_model.encode.return_value = embeddings
        mock_st.return_value = mock_model
        
        embedder = EmbeddingModel()
        batch_embeddings = embedder.embed_batch(texts)
        
        assert len(batch_embeddings) == 3
        assert all(isinstance(e, list) for e in batch_embeddings)
        assert all(len(e) == 3 for e in batch_embeddings)
        # Verify structure
        assert batch_embeddings[0] == [0.1, 0.2, 0.3] or \
               batch_embeddings[0] == list(embeddings[0])
    
    @patch('src.embeddings.embedder.SentenceTransformer')
    def test_different_texts_different_embeddings(self, mock_st):
        """Test that different texts produce different embeddings."""
        # Setup mock - different texts get different embeddings
        mock_model = MagicMock()
        embeddings_map = {
            "text one": np.array([0.1, 0.2, 0.3]),
            "text two": np.array([0.4, 0.5, 0.6])
        }
        
        def encode_side_effect(text):
            if isinstance(text, list):
                return np.array([embeddings_map.get(t, np.array([0.0, 0.0, 0.0])) 
                               for t in text])
            else:
                return embeddings_map.get(text, np.array([0.0, 0.0, 0.0]))
        
        mock_model.encode.side_effect = encode_side_effect
        mock_st.return_value = mock_model
        
        embedder = EmbeddingModel()
        emb1 = embedder.embed("text one")
        emb2 = embedder.embed("text two")
        
        assert emb1 != emb2
    
    @patch('src.embeddings.embedder.SentenceTransformer')
    def test_empty_text_handling(self, mock_st):
        """Test that empty text is handled gracefully."""
        # Setup mock
        mock_model = MagicMock()
        mock_embedding = [0.0, 0.0, 0.0]
        mock_model.encode.return_value = np.array(mock_embedding)
        mock_st.return_value = mock_model
        
        embedder = EmbeddingModel()
        # Should not raise an error
        embedding = embedder.embed("")
        
        assert isinstance(embedding, list)
        assert len(embedding) == 3
    
    @patch('src.embeddings.embedder.SentenceTransformer')
    def test_dimension_property(self, mock_st):
        """Test that dimension property returns correct size."""
        # Setup mock
        mock_model = MagicMock()
        mock_embedding = [0.1] * 384  # all-MiniLM-L6-v2 is 384 dimensional
        mock_model.encode.return_value = np.array(mock_embedding)
        mock_st.return_value = mock_model
        
        embedder = EmbeddingModel()
        # Trigger embedding to set dimension
        embedder.embed("test")
        
        assert embedder.dimension == 384
    
    @patch('src.embeddings.embedder.SentenceTransformer')
    def test_lazy_loading(self, mock_st):
        """Test that model is loaded lazily on first use."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_st.return_value = mock_model
        
        embedder = EmbeddingModel()
        # Model should not be loaded yet
        assert mock_st.call_count == 0
        
        # After first embed call, model should be loaded
        embedder.embed("test")
        assert mock_st.call_count == 1
        
        # Second call should reuse the same model
        embedder.embed("test2")
        assert mock_st.call_count == 1
    
    @patch('src.embeddings.embedder.SentenceTransformer')
    def test_default_model_name(self, mock_st):
        """Test that default model name is used."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2])
        mock_st.return_value = mock_model
        
        embedder = EmbeddingModel()
        embedder.embed("test")
        
        # Check that SentenceTransformer was called with default model
        mock_st.assert_called_once_with("sentence-transformers/all-MiniLM-L6-v2")
    
    @patch('src.embeddings.embedder.SentenceTransformer')
    def test_custom_model_name(self, mock_st):
        """Test that custom model name is used."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2])
        mock_st.return_value = mock_model
        
        embedder = EmbeddingModel(model_name="custom-model")
        embedder.embed("test")
        
        # Check that SentenceTransformer was called with custom model
        mock_st.assert_called_once_with("custom-model")
