"""
Embedding generation service using sentence-transformers.
"""

import logging
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating sentence embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding service.

        Args:
            model_name: Name of the sentence-transformers model to use
        """
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        logger.info(
            "Embedding model loaded successfully. Embedding dimension: %s",
            self.model.get_sentence_embedding_dimension()
        )

    def generate_embeddings(
        self,
        sentences: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for a list of sentences.

        Args:
            sentences: List of sentence strings
            batch_size: Batch size for processing (larger = faster but more memory)
            show_progress_bar: Whether to show progress bar during encoding

        Returns:
            numpy array of shape (len(sentences), embedding_dim)
        """
        if not sentences:
            raise ValueError("Cannot generate embeddings for empty sentence list")

        logger.info(
            "Generating embeddings for %s sentences with batch size %s",
            len(sentences), batch_size
        )

        try:
            embeddings = self.model.encode(
                sentences,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for cosine similarity
            )
            logger.info(
                "Successfully generated embeddings with shape: %s",
                embeddings.shape
            )
            return embeddings

        except Exception as e:
            logger.error("Error generating embeddings: %s", str(e), exc_info=True)
            raise

    def generate_embeddings_with_metadata(
        self,
        sentence_data: List[Dict[str, str]],
        batch_size: int = 32
    ) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for sentences with metadata preservation.

        Args:
            sentence_data: List of dicts with 'sentence' and 'id' keys
            batch_size: Batch size for processing

        Returns:
            Dictionary mapping sentence IDs to their embeddings
        """
        if not sentence_data:
            raise ValueError("Cannot generate embeddings for empty data")

        sentences = [item['sentence'] for item in sentence_data]
        sentence_ids = [item['id'] for item in sentence_data]

        embeddings = self.generate_embeddings(sentences, batch_size=batch_size)

        # Create mapping of ID to embedding
        # Note: Multiple sentences may share the same ID (from same comment)
        # We'll return all embeddings and handle deduplication at clustering level
        result = {}
        for idx, (sentence_id, embedding) in enumerate(zip(sentence_ids, embeddings)):
            # Use index as unique key to preserve all sentences
            result[f"{sentence_id}_{idx}"] = {
                'id': sentence_id,
                'sentence': sentences[idx],
                'embedding': embedding
            }

        logger.info(
            "Generated embeddings for %s sentence instances",
            len(result)
        )
        return result
