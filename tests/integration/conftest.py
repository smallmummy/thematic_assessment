"""
Shared fixtures for integration tests.

This module provides reusable fixtures for integration testing.
Models are loaded once per session for performance.
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch

from src.services.embeddings import EmbeddingService
from src.services.sentiment import SentimentService
from src.services.clustering import ClusteringService
from src.services.insights import InsightService


@pytest.fixture(scope="session")
def embedding_service():
    """
    Real embedding service using all-MiniLM-L6-v2 model.
    Loaded once per test session for performance.
    """
    return EmbeddingService(model_name="all-MiniLM-L6-v2")


@pytest.fixture(scope="session")
def sentiment_service():
    """
    Real sentiment service using BERTweet model.
    Loaded once per test session for performance.
    """
    return SentimentService(
        model_name="finiteautomata/bertweet-base-sentiment-analysis",
        max_text_length=512
    )


@pytest.fixture
def clustering_service():
    """Real clustering service with default parameters."""
    return ClusteringService(
        min_cluster_size=3,
        min_samples=1,
        metric='euclidean'
    )


@pytest.fixture
def mock_openai_client():
    """
    Mock OpenAI client for insight generation.
    This is the ONLY external service we mock in integration tests.
    """

    # Mock async client (actually used)
    mock_async = Mock()

    async def mock_create(**kwargs):
        """Generate realistic responses based on the prompt."""
        messages = kwargs.get('messages', [])
        prompt = messages[-1]['content'] if messages else ''

        # Determine if this is standalone or comparative based on prompt
        if 'similarities' in prompt.lower() and 'differences' in prompt.lower():
            # Comparative analysis
            response_content = json.dumps({
                'title': 'Comparative Theme Analysis',
                'similarities': [
                    'Both datasets mention similar topics',
                    'Common concerns across both groups'
                ],
                'differences': [
                    'Baseline shows more positive sentiment',
                    'Comparison includes additional concerns'
                ]
            })
        else:
            # Standalone analysis
            # Extract theme from prompt if possible
            theme = 'General'
            if 'theme:' in prompt.lower():
                # Simple extraction (in real prompt it's more structured)
                theme = 'Customer Feedback'

            response_content = json.dumps({
                'title': f'{theme} Insights',
                'insights': [
                    'Key pattern identified in the data',
                    'Significant trend observed across sentences'
                ]
            })

        return Mock(choices=[Mock(message=Mock(content=response_content))])

    mock_async.chat.completions.create = AsyncMock(side_effect=mock_create)

    return mock_async


@pytest.fixture
def insight_service_with_mock(mock_openai_client):
    """
    Insight service with mocked OpenAI client.
    Uses session-scoped mock to avoid API calls.
    """

    mock_async = mock_openai_client

    with patch('src.services.insights.AsyncOpenAI', return_value=mock_async):
        service = InsightService(
            api_key="test-key",
            model="gpt-4o",
            max_sentences_for_insights=20,
            max_sentences_for_comparison=15,
            max_title_length=100
        )
        yield service


@pytest.fixture
def sample_quality_feedback():
    """Sample sentences about product quality (should cluster together)."""
    return [
        {'sentence': 'The product quality is excellent and outstanding', 'id': 'q1'},
        {'sentence': 'Very satisfied with the build quality and craftsmanship', 'id': 'q2'},
        {'sentence': 'Great materials used in construction and manufacturing', 'id': 'q3'},
        {'sentence': 'Quality exceeds expectations and is superior', 'id': 'q4'},
        {'sentence': 'Outstanding quality throughout the entire product', 'id': 'q5'},
        {'sentence': 'Superior build quality and excellent materials', 'id': 'q6'},
    ]


@pytest.fixture
def sample_price_feedback():
    """Sample sentences about pricing (should cluster together)."""
    return [
        {'sentence': 'The price is too high for this quality', 'id': 'p1'},
        {'sentence': 'Not worth the money', 'id': 'p2'},
        {'sentence': 'Overpriced for what you get', 'id': 'p3'},
        {'sentence': 'Too expensive compared to competitors', 'id': 'p4'},
    ]


@pytest.fixture
def sample_shipping_feedback():
    """Sample sentences about shipping (should cluster together)."""
    return [
        {'sentence': 'Delivery was very fast', 'id': 's1'},
        {'sentence': 'Shipping speed exceeded expectations', 'id': 's2'},
        {'sentence': 'Received the package quickly', 'id': 's3'},
        {'sentence': 'Quick delivery service', 'id': 's4'},
    ]


@pytest.fixture
def sample_mixed_feedback(sample_quality_feedback, sample_price_feedback, sample_shipping_feedback):
    """Mixed feedback that should form 3 distinct clusters."""
    return sample_quality_feedback + sample_price_feedback + sample_shipping_feedback


@pytest.fixture
def sample_comparative_baseline():
    """Baseline dataset for comparative analysis."""
    return [
        {'sentence': 'The old interface is intuitive', 'id': 'b1'},
        {'sentence': 'Navigation is straightforward', 'id': 'b2'},
        {'sentence': 'Design is clean', 'id': 'b3'},
        {'sentence': 'Loading times are acceptable', 'id': 'b4'},
        {'sentence': 'Some features are hard to find', 'id': 'b5'},
        {'sentence': 'Search function works well', 'id': 'b6'},
    ]


@pytest.fixture
def sample_comparative_comparison():
    """Comparison dataset for comparative analysis."""
    return [
        {'sentence': 'The new interface is much better', 'id': 'c1'},
        {'sentence': 'Navigation has improved significantly', 'id': 'c2'},
        {'sentence': 'Love the updated design', 'id': 'c3'},
        {'sentence': 'Much faster loading times now', 'id': 'c4'},
        {'sentence': 'All features are easier to access', 'id': 'c5'},
        {'sentence': 'Search is more powerful', 'id': 'c6'},
    ]