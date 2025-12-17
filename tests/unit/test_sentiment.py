"""
Unit tests for SentimentService.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.services.sentiment import SentimentService


class TestSentimentService:
    """Test SentimentService class."""

    @patch('src.services.sentiment.pipeline')
    def test_initialization(self, mock_pipeline):
        """Test service initialization."""
        mock_classifier = Mock()
        mock_pipeline.return_value = mock_classifier

        service = SentimentService(
            model_name="test-model",
            max_text_length=256
        )

        mock_pipeline.assert_called_once_with(
            "sentiment-analysis",
            model="test-model",
            top_k=None
        )
        assert service.model_name == "test-model"
        assert service.max_text_length == 256
        assert service.classifier == mock_classifier

    @patch('src.services.sentiment.pipeline')
    def test_initialization_defaults(self, mock_pipeline):
        """Test service initialization with defaults."""
        mock_pipeline.return_value = Mock()

        service = SentimentService()

        assert service.model_name == "finiteautomata/bertweet-base-sentiment-analysis"
        assert service.max_text_length == 512

    @patch('src.services.sentiment.pipeline')
    def test_analyze_batch_sentiment_success(self, mock_pipeline):
        """Test successful batch sentiment analysis."""
        # Setup mock classifier
        mock_classifier = Mock()
        mock_classifier.return_value = [
            [
                {'label': 'POS', 'score': 0.8},
                {'label': 'NEU', 'score': 0.15},
                {'label': 'NEG', 'score': 0.05}
            ],
            [
                {'label': 'NEG', 'score': 0.9},
                {'label': 'NEU', 'score': 0.08},
                {'label': 'POS', 'score': 0.02}
            ]
        ]
        mock_pipeline.return_value = mock_classifier

        service = SentimentService()
        texts = ["This is great!", "This is terrible!"]

        result = service.analyze_batch_sentiment(texts, batch_size=8)

        # Verify results
        assert len(result) == 2

        # First result should be positive
        assert result[0]['label'] == 'positive'
        assert result[0]['positive'] == 0.8
        assert result[0]['neutral'] == 0.15
        assert result[0]['negative'] == 0.05

        # Second result should be negative
        assert result[1]['label'] == 'negative'
        assert result[1]['positive'] == 0.02
        assert result[1]['neutral'] == 0.08
        assert result[1]['negative'] == 0.9

    @patch('src.services.sentiment.pipeline')
    def test_analyze_batch_sentiment_neutral(self, mock_pipeline):
        """Test neutral sentiment classification."""
        mock_classifier = Mock()
        mock_classifier.return_value = [
            [
                {'label': 'NEU', 'score': 0.7},
                {'label': 'POS', 'score': 0.2},
                {'label': 'NEG', 'score': 0.1}
            ]
        ]
        mock_pipeline.return_value = mock_classifier

        service = SentimentService()
        result = service.analyze_batch_sentiment(["Neutral text"])

        assert result[0]['label'] == 'neutral'
        assert result[0]['neutral'] == 0.7

    @patch('src.services.sentiment.pipeline')
    def test_analyze_batch_sentiment_empty_list(self, mock_pipeline):
        """Test that empty list raises ValueError."""
        mock_pipeline.return_value = Mock()
        service = SentimentService()

        with pytest.raises(ValueError) as exc_info:
            service.analyze_batch_sentiment([])
        assert "Cannot analyze sentiment of empty text list" in str(exc_info.value)

    @patch('src.services.sentiment.pipeline')
    def test_analyze_batch_sentiment_truncation(self, mock_pipeline):
        """Test text truncation to max length."""
        mock_classifier = Mock()
        mock_classifier.return_value = [
            [
                {'label': 'POS', 'score': 0.8},
                {'label': 'NEU', 'score': 0.15},
                {'label': 'NEG', 'score': 0.05}
            ]
        ]
        mock_pipeline.return_value = mock_classifier

        service = SentimentService(max_text_length=10)
        long_text = "A" * 100  # 100 characters

        service.analyze_batch_sentiment([long_text])

        # Verify truncation occurred
        call_args = mock_classifier.call_args[0][0]
        assert len(call_args[0]) == 10

    @patch('src.services.sentiment.pipeline')
    def test_analyze_batch_sentiment_batching(self, mock_pipeline):
        """Test batching logic."""
        mock_classifier = Mock()
        # First batch (2 items)
        mock_classifier.side_effect = [
            [
                [{'label': 'POS', 'score': 0.8}, {'label': 'NEU', 'score': 0.15}, {'label': 'NEG', 'score': 0.05}],
                [{'label': 'NEG', 'score': 0.7}, {'label': 'NEU', 'score': 0.2}, {'label': 'POS', 'score': 0.1}]
            ],
            # Second batch (1 item)
            [
                [{'label': 'NEU', 'score': 0.6}, {'label': 'POS', 'score': 0.3}, {'label': 'NEG', 'score': 0.1}]
            ]
        ]
        mock_pipeline.return_value = mock_classifier

        service = SentimentService()
        texts = ["Text 1", "Text 2", "Text 3"]

        result = service.analyze_batch_sentiment(texts, batch_size=2)

        # Should have been called twice (2 batches)
        assert mock_classifier.call_count == 2
        assert len(result) == 3

    @patch('src.services.sentiment.pipeline')
    def test_aggregate_cluster_sentiment_majority_positive(self, mock_pipeline):
        """Test cluster sentiment aggregation with positive majority."""
        mock_classifier = Mock()
        mock_classifier.return_value = [
            [{'label': 'POS', 'score': 0.8}, {'label': 'NEU', 'score': 0.15}, {'label': 'NEG', 'score': 0.05}],
            [{'label': 'POS', 'score': 0.7}, {'label': 'NEU', 'score': 0.2}, {'label': 'NEG', 'score': 0.1}],
            [{'label': 'NEG', 'score': 0.6}, {'label': 'NEU', 'score': 0.3}, {'label': 'POS', 'score': 0.1}]
        ]
        mock_pipeline.return_value = mock_classifier

        service = SentimentService()
        sentences = ["Great!", "Excellent!", "Bad"]

        result = service.aggregate_cluster_sentiment(sentences)

        assert result == 'positive'  # 2 positive, 1 negative

    @patch('src.services.sentiment.pipeline')
    def test_aggregate_cluster_sentiment_tie_breaker(self, mock_pipeline):
        """Test cluster sentiment aggregation with tie-breaking."""
        mock_classifier = Mock()
        # Equal counts but different average scores
        mock_classifier.return_value = [
            [{'label': 'POS', 'score': 0.9}, {'label': 'NEU', 'score': 0.08}, {'label': 'NEG', 'score': 0.02}],
            [{'label': 'NEG', 'score': 0.6}, {'label': 'NEU', 'score': 0.3}, {'label': 'POS', 'score': 0.1}]
        ]
        mock_pipeline.return_value = mock_classifier

        service = SentimentService()
        sentences = ["Very positive!", "Somewhat negative"]

        result = service.aggregate_cluster_sentiment(sentences)

        # Should use average scores as tie-breaker
        # Positive: (0.9 + 0.1) / 2 = 0.5
        # Negative: (0.02 + 0.6) / 2 = 0.31
        # Neutral: (0.08 + 0.3) / 2 = 0.19
        assert result == 'positive'

    @patch('src.services.sentiment.pipeline')
    def test_aggregate_cluster_sentiment_all_neutral(self, mock_pipeline):
        """Test cluster sentiment with all neutral."""
        mock_classifier = Mock()
        mock_classifier.return_value = [
            [{'label': 'NEU', 'score': 0.8}, {'label': 'POS', 'score': 0.1}, {'label': 'NEG', 'score': 0.1}],
            [{'label': 'NEU', 'score': 0.7}, {'label': 'POS', 'score': 0.15}, {'label': 'NEG', 'score': 0.15}]
        ]
        mock_pipeline.return_value = mock_classifier

        service = SentimentService()
        sentences = ["Okay", "It's fine"]

        result = service.aggregate_cluster_sentiment(sentences)

        assert result == 'neutral'

    @patch('src.services.sentiment.pipeline')
    def test_aggregate_cluster_sentiment_empty_list(self, mock_pipeline):
        """Test that empty list raises ValueError."""
        mock_pipeline.return_value = Mock()
        service = SentimentService()

        with pytest.raises(ValueError) as exc_info:
            service.aggregate_cluster_sentiment([])
        assert "Cannot aggregate sentiment for empty sentence list" in str(exc_info.value)

    @patch('src.services.sentiment.pipeline')
    def test_label_mapping(self, mock_pipeline):
        """Test label mapping from model output."""
        mock_classifier = Mock()
        mock_classifier.return_value = [
            [
                {'label': 'pos', 'score': 0.5},  # Lowercase
                {'label': 'neu', 'score': 0.3},
                {'label': 'neg', 'score': 0.2}
            ]
        ]
        mock_pipeline.return_value = mock_classifier

        service = SentimentService()
        result = service.analyze_batch_sentiment(["Test"])

        # Verify mapping works with lowercase
        assert result[0]['positive'] == 0.5
        assert result[0]['neutral'] == 0.3
        assert result[0]['negative'] == 0.2

    def test_sentiment_accuracy(self):
        """Test sentiment classification accuracy on known examples."""
        service = SentimentService(
            model_name="distilbert-base-uncased-finetuned-sst-2-english",
            max_text_length=256
        )

        test_cases = [
            # Clearly positive
            ("I love this product!", "positive"),
            ("Excellent quality and fast shipping", "positive"),
            ("Best purchase ever", "positive"),
            ("Overpriced but decent quality", "positive"),

            # Clearly negative
            ("I hate this product", "negative"),
            ("Terrible quality, waste of money", "negative"),
            ("Worst experience ever", "negative"),

            # Neutral
            ("It works as expected", "neutral"),
            ("The product arrived on time", "neutral"),
            ("The quality is good but price is too high", "negative")
        ]

        correct = 0
        for text, expected in test_cases:
            result = service.analyze_batch_sentiment([text], batch_size=8)
            if result[0]['label'] == expected:
                correct += 1
            else:
                print(f"MISS: '{text}' -> {result} (expected {expected})")

        accuracy = correct / len(test_cases)
        print(f"Accuracy: {accuracy:.1%} ({correct}/{len(test_cases)})")

        assert accuracy >= 0.8  # Require 80% accuracy
