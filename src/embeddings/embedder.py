"""EmbeddingModel for generating vector embeddings from text."""

from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel:
    """Generates vector embeddings for text using SentenceTransformers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize with a sentence-transformers model name.

        Args:
            model_name: Name of the SentenceTransformer model to use.
                       Defaults to all-MiniLM-L6-v2 (384 dimensions).
        """
        self.model_name = model_name
        self._model = None
        self._dimension = None

    def _load_model(self):
        """Load the model lazily on first use."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            List of floats representing the embedding vector.
        """
        self._load_model()
        embedding = self._model.encode(text)

        # Convert numpy array to list of floats
        result = embedding.tolist() if isinstance(embedding, np.ndarray) else list(embedding)

        # Cache dimension on first use
        if self._dimension is None:
            self._dimension = len(result)

        return result

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors (each a list of floats).
        """
        self._load_model()
        embeddings = self._model.encode(texts)

        # Convert numpy array to list of lists
        if isinstance(embeddings, np.ndarray):
            result = embeddings.tolist()
        else:
            result = [list(e) if isinstance(e, np.ndarray) else list(e) for e in embeddings]

        # Cache dimension on first use
        if self._dimension is None and len(result) > 0:
            self._dimension = len(result[0])

        return result

    @property
    def dimension(self) -> int:
        """Return embedding dimension size.

        Returns:
            The dimension of the embedding vectors.
        """
        # Ensure model is loaded to get dimension
        if self._dimension is None:
            self._load_model()
            # Generate a dummy embedding to determine dimension
            dummy = self._model.encode("")
            self._dimension = len(dummy)

        return self._dimension
