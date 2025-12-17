"""
Unit tests for EmbeddingService.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.services.embeddings import EmbeddingService


class TestEmbeddingService:
    """Test EmbeddingService class."""

    @patch('src.services.embeddings.SentenceTransformer')
    def test_initialization(self, mock_transformer):
        """Test service initialization."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model

        service = EmbeddingService(model_name="test-model")

        mock_transformer.assert_called_once_with("test-model")
        assert service.model_name == "test-model"
        assert service.model == mock_model

    @patch('src.services.embeddings.SentenceTransformer')
    def test_generate_embeddings_success(self, mock_transformer):
        """Test successful embedding generation."""
        # Setup mock
        mock_model = Mock()
        mock_embeddings = np.random.rand(3, 384)
        mock_model.encode.return_value = mock_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model

        service = EmbeddingService()
        sentences = ["Test 1", "Test 2", "Test 3"]

        result = service.generate_embeddings(sentences, batch_size=16)

        # Verify
        assert np.array_equal(result, mock_embeddings)
        mock_model.encode.assert_called_once()
        call_args = mock_model.encode.call_args
        assert call_args[0][0] == sentences
        assert call_args[1]['batch_size'] == 16
        assert call_args[1]['convert_to_numpy'] is True
        assert call_args[1]['normalize_embeddings'] is True

    @patch('src.services.embeddings.SentenceTransformer')
    def test_generate_embeddings_empty_list(self, mock_transformer):
        """Test that empty list raises ValueError."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model

        service = EmbeddingService()

        with pytest.raises(ValueError) as exc_info:
            service.generate_embeddings([])
        assert "Cannot generate embeddings for empty sentence list" in str(exc_info.value)

    @patch('src.services.embeddings.SentenceTransformer')
    def test_generate_embeddings_error_handling(self, mock_transformer):
        """Test error handling during embedding generation."""
        mock_model = Mock()
        mock_model.encode.side_effect = RuntimeError("Model error")
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model

        service = EmbeddingService()

        with pytest.raises(RuntimeError) as exc_info:
            service.generate_embeddings(["Test"])
        assert "Model error" in str(exc_info.value)

    @patch('src.services.embeddings.SentenceTransformer')
    def test_generate_embeddings_with_metadata_success(self, mock_transformer):
        """Test embedding generation with metadata."""
        # Setup mock
        mock_model = Mock()
        mock_embeddings = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ])
        mock_model.encode.return_value = mock_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 3
        mock_transformer.return_value = mock_model

        service = EmbeddingService()
        sentence_data = [
            {'sentence': 'Test 1', 'id': 'id1'},
            {'sentence': 'Test 2', 'id': 'id2'},
            {'sentence': 'Test 3', 'id': 'id1'}  # Duplicate ID
        ]

        result = service.generate_embeddings_with_metadata(sentence_data)

        # Verify structure
        assert len(result) == 3
        assert 'id1_0' in result
        assert 'id2_1' in result
        assert 'id1_2' in result

        # Verify first entry
        assert result['id1_0']['id'] == 'id1'
        assert result['id1_0']['sentence'] == 'Test 1'
        assert np.array_equal(result['id1_0']['embedding'], mock_embeddings[0])

        # Verify duplicate ID handling
        assert result['id1_2']['id'] == 'id1'
        assert result['id1_2']['sentence'] == 'Test 3'
        assert np.array_equal(result['id1_2']['embedding'], mock_embeddings[2])

    @patch('src.services.embeddings.SentenceTransformer')
    def test_generate_embeddings_with_metadata_empty_data(self, mock_transformer):
        """Test that empty data raises ValueError."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model

        service = EmbeddingService()

        with pytest.raises(ValueError) as exc_info:
            service.generate_embeddings_with_metadata([])
        assert "Cannot generate embeddings for empty data" in str(exc_info.value)

    @patch('src.services.embeddings.SentenceTransformer')
    def test_generate_embeddings_with_custom_batch_size(self, mock_transformer):
        """Test embedding generation with custom batch size."""
        mock_model = Mock()
        mock_embeddings = np.random.rand(5, 384)
        mock_model.encode.return_value = mock_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model

        service = EmbeddingService()
        sentence_data = [
            {'sentence': f'Test {i}', 'id': str(i)} for i in range(5)
        ]

        result = service.generate_embeddings_with_metadata(sentence_data, batch_size=64)

        # Verify batch size was passed through
        call_args = mock_model.encode.call_args
        assert call_args[1]['batch_size'] == 64

    @patch('src.services.embeddings.SentenceTransformer')
    def test_embedding_normalization(self, mock_transformer):
        """Test that embeddings are normalized."""
        mock_model = Mock()
        # Create non-normalized embeddings
        mock_embeddings = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mock_model.encode.return_value = mock_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 3
        mock_transformer.return_value = mock_model

        service = EmbeddingService()
        sentences = ["Test 1", "Test 2"]

        service.generate_embeddings(sentences)

        # Verify normalize_embeddings flag was set
        call_args = mock_model.encode.call_args
        assert call_args[1]['normalize_embeddings'] is True
