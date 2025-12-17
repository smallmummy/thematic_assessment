"""
Integration tests for Lambda handler with real analyzer.

These tests use the real TextAnalyzer with real ML models,
only mocking the OpenAI API.
"""

import json
import os
import pytest
from unittest.mock import patch, Mock, AsyncMock

from lambda_handler import lambda_handler


@pytest.mark.integration
class TestLambdaHandlerIntegration:
    """Integration tests for Lambda handler with real services."""

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    @patch('src.services.insights.AsyncOpenAI')
    @patch('src.services.insights.OpenAI')
    def test_lambda_standalone_full_pipeline(
        self,
        mock_openai,
        mock_async_openai,
        sample_quality_feedback
    ):
        """
        Test Lambda handler with real TextAnalyzer for standalone analysis.

        Only mocks OpenAI API, everything else is real.
        """
        # Setup OpenAI mocks
        async def mock_create(**kwargs):
            return Mock(choices=[Mock(message=Mock(content=json.dumps({
                'title': 'Quality Feedback Analysis',
                'insights': ['High quality standards', 'Customer satisfaction evident']
            })))])

        mock_async = Mock()
        mock_async.chat.completions.create = AsyncMock(side_effect=mock_create)
        mock_async_openai.return_value = mock_async

        mock_sync = Mock()
        mock_sync.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="Test Title"))]
        )
        mock_openai.return_value = mock_sync

        # Create Lambda event
        event = {
            'body': json.dumps({
                'surveyTitle': 'Product Quality Survey',
                'theme': 'Customer Satisfaction',
                'baseline': sample_quality_feedback
            })
        }

        # Invoke Lambda handler
        response = lambda_handler(event, None)

        # Verify response structure
        assert response['statusCode'] == 200
        assert 'Content-Type' in response['headers']
        assert response['headers']['Access-Control-Allow-Origin'] == '*'

        # Parse and validate body
        body = json.loads(response['body'])
        assert 'clusters' in body
        assert isinstance(body['clusters'], list)

        # Validate cluster structure
        for cluster in body['clusters']:
            assert 'title' in cluster
            assert 'sentiment' in cluster
            assert 'sentences' in cluster
            assert 'keyInsights' in cluster
            assert cluster['sentiment'] in ['positive', 'negative', 'neutral']
            assert isinstance(cluster['sentences'], list)
            assert isinstance(cluster['keyInsights'], list)
            assert len(cluster['keyInsights']) >= 2
            assert len(cluster['keyInsights']) <= 3

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    @patch('src.services.insights.AsyncOpenAI')
    @patch('src.services.insights.OpenAI')
    def test_lambda_comparative_full_pipeline(
        self,
        mock_openai,
        mock_async_openai,
        sample_comparative_baseline,
        sample_comparative_comparison
    ):
        """
        Test Lambda handler with real TextAnalyzer for comparative analysis.

        Only mocks OpenAI API, everything else is real.
        """
        # Setup OpenAI mocks
        async def mock_create(**kwargs):
            return Mock(choices=[Mock(message=Mock(content=json.dumps({
                'title': 'UI Comparison',
                'similarities': ['Both focus on usability', 'Navigation is key concern'],
                'differences': ['New version faster', 'Improved feature accessibility']
            })))])

        mock_async = Mock()
        mock_async.chat.completions.create = AsyncMock(side_effect=mock_create)
        mock_async_openai.return_value = mock_async

        mock_sync = Mock()
        mock_sync.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="Test Title"))]
        )
        mock_openai.return_value = mock_sync

        # Create Lambda event
        event = {
            'body': json.dumps({
                'surveyTitle': 'UI Comparison Study',
                'theme': 'User Experience',
                'baseline': sample_comparative_baseline,
                'comparison': sample_comparative_comparison
            })
        }

        # Invoke Lambda handler
        response = lambda_handler(event, None)

        # Verify response structure
        assert response['statusCode'] == 200
        assert 'Content-Type' in response['headers']

        # Parse and validate body
        body = json.loads(response['body'])
        assert 'clusters' in body
        assert isinstance(body['clusters'], list)

        # Validate comparative cluster structure
        for cluster in body['clusters']:
            assert 'title' in cluster
            assert 'sentiment' in cluster
            assert 'baselineSentences' in cluster
            assert 'comparisonSentences' in cluster
            assert 'keySimilarities' in cluster
            assert 'keyDifferences' in cluster
            assert isinstance(cluster['baselineSentences'], list)
            assert isinstance(cluster['comparisonSentences'], list)
            assert len(cluster['keySimilarities']) >= 2
            assert len(cluster['keyDifferences']) >= 2

    @patch.dict('os.environ', {
        'OPENAI_API_KEY': 'test-key',
        'MIN_CLUSTER_SIZE': '2',
        'MAX_CLUSTERS': '10'
    })
    @patch('src.services.insights.AsyncOpenAI')
    @patch('src.services.insights.OpenAI')
    def test_lambda_with_custom_config(
        self,
        mock_openai,
        mock_async_openai,
        sample_quality_feedback
    ):
        """
        Test Lambda handler respects environment variable configuration.
        """
        # Setup OpenAI mocks
        async def mock_create(**kwargs):
            return Mock(choices=[Mock(message=Mock(content=json.dumps({
                'title': 'Test',
                'insights': ['Insight 1', 'Insight 2']
            })))])

        mock_async = Mock()
        mock_async.chat.completions.create = AsyncMock(side_effect=mock_create)
        mock_async_openai.return_value = mock_async

        mock_sync = Mock()
        mock_openai.return_value = mock_sync

        event = {
            'body': json.dumps({
                'surveyTitle': 'Config Test',
                'theme': 'Test',
                'baseline': sample_quality_feedback[:3]  # Small dataset
            })
        }

        response = lambda_handler(event, None)

        # Should succeed with custom config
        assert response['statusCode'] == 200

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    @patch('src.services.insights.AsyncOpenAI')
    @patch('src.services.insights.OpenAI')
    def test_lambda_error_handling_integration(
        self,
        mock_openai,
        mock_async_openai
    ):
        """
        Test Lambda error handling with real validation.
        """
        # Setup mocks (won't be called due to validation error)
        mock_async_openai.return_value = Mock()
        mock_openai.return_value = Mock()

        # Invalid request - empty baseline
        event = {
            'body': json.dumps({
                'surveyTitle': 'Test',
                'theme': 'Test',
                'baseline': []  # Invalid - empty
            })
        }

        response = lambda_handler(event, None)

        # Should return 400 error
        assert response['statusCode'] == 400
        body = json.loads(response['body'])
        assert 'error' in body
        assert body['error']['code'] == 'VALIDATION_ERROR'

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    @patch('src.services.insights.AsyncOpenAI')
    @patch('src.services.insights.OpenAI')
    def test_lambda_json_parsing_integration(
        self,
        mock_openai,
        mock_async_openai
    ):
        """
        Test Lambda JSON parsing with real handler.
        """
        # Setup mocks
        mock_async_openai.return_value = Mock()
        mock_openai.return_value = Mock()

        # Invalid JSON
        event = {
            'body': 'invalid json {'
        }

        response = lambda_handler(event, None)

        # Should return 400 error
        assert response['statusCode'] == 400
        body = json.loads(response['body'])
        assert 'error' in body
        assert body['error']['code'] == 'INVALID_JSON'

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    @patch('src.services.insights.AsyncOpenAI')
    @patch('src.services.insights.OpenAI')
    def test_lambda_real_clustering_behavior(
        self,
        mock_openai,
        mock_async_openai
    ):
        """
        Test that Lambda uses real clustering and returns realistic results.

        Uses diverse sentences that should form multiple clusters.
        """
        # Setup OpenAI mocks
        async def mock_create(**kwargs):
            return Mock(choices=[Mock(message=Mock(content=json.dumps({
                'title': 'Cluster Analysis',
                'insights': ['Pattern detected', 'Trend observed']
            })))])

        mock_async = Mock()
        mock_async.chat.completions.create = AsyncMock(side_effect=mock_create)
        mock_async_openai.return_value = mock_async
        mock_openai.return_value = Mock()

        # Diverse sentences with stronger similarity within themes
        # More sentences per theme for better clustering
        event = {
            'body': json.dumps({
                'surveyTitle': 'Multi-Topic Survey',
                'theme': 'Product Feedback',
                'baseline': [
                    # Quality theme (4 sentences)
                    {'sentence': 'Quality is excellent and outstanding', 'id': 'q1'},
                    {'sentence': 'Build quality is great and impressive', 'id': 'q2'},
                    {'sentence': 'Product quality exceeds expectations', 'id': 'q3'},
                    {'sentence': 'Superior quality and craftsmanship', 'id': 'q4'},
                    # Price theme (4 sentences)
                    {'sentence': 'Price is too high and expensive', 'id': 'p1'},
                    {'sentence': 'Too expensive for the budget', 'id': 'p2'},
                    {'sentence': 'Overpriced and costs too much', 'id': 'p3'},
                    {'sentence': 'Pricing is unreasonable', 'id': 'p4'},
                    # Shipping theme (4 sentences)
                    {'sentence': 'Fast delivery and quick shipping', 'id': 's1'},
                    {'sentence': 'Quick shipping and fast arrival', 'id': 's2'},
                    {'sentence': 'Delivery was very fast', 'id': 's3'},
                    {'sentence': 'Shipping speed was excellent', 'id': 's4'},
                ]
            })
        }

        response = lambda_handler(event, None)

        assert response['statusCode'] == 200
        body = json.loads(response['body'])

        # Real clustering should create clusters for diverse topics
        # With 12 sentences across 3 themes, should form at least 1 cluster
        assert len(body['clusters']) >= 1, "Should create at least one cluster with sufficient data"

        # All input IDs should appear in output
        all_output_ids = []
        for cluster in body['clusters']:
            all_output_ids.extend(cluster['sentences'])

        input_ids = {'q1', 'q2', 'q3', 'q4', 'p1', 'p2', 'p3', 'p4', 's1', 's2', 's3', 's4'}
        output_ids_set = set(all_output_ids)

        # Most IDs should be clustered (some might be noise)
        overlap = input_ids & output_ids_set
        assert len(overlap) >= len(input_ids) * 0.5, \
            "At least 50% of sentences should be clustered"