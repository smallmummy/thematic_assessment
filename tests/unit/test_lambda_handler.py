"""
Unit tests for Lambda handler.
"""

import json
import sys
import pytest
from unittest.mock import Mock, patch

from lambda_handler import (
    lambda_handler,
    handle_standalone_analysis,
    handle_comparative_analysis,
    create_error_response
)


class TestLambdaHandler:
    """Test Lambda handler function."""

    def setup_method(self):
        """Clear module cache before each test to ensure clean imports."""
        # Clear the global TextAnalyzer cache in lambda_handler module
        # This ensures each test gets a fresh analyzer instance (not cached from previous tests)
        import lambda_handler
        lambda_handler._text_analyzer_cache = None
        lambda_handler._openai_api_key_cache = None

        # Remove the text_analyzer module from cache to force re-import
        # This ensures that @patch decorators work correctly with delayed imports
        if 'src.services.text_analyzer' in sys.modules:
            del sys.modules['src.services.text_analyzer']

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    @patch('src.services.text_analyzer.TextAnalyzer')
    def test_lambda_handler_standalone_success(self, mock_analyzer_class):
        """Test successful standalone analysis via Lambda."""
        # Setup mock
        mock_analyzer = Mock()
        mock_result = Mock()
        mock_result.clusters = [Mock()]  # Add clusters attribute for len() call
        mock_result.model_dump.return_value = {
            'clusters': [
                {
                    'title': 'Test Cluster',
                    'sentiment': 'positive',
                    'sentences': ['1', '2'],
                    'keyInsights': ['Insight 1', 'Insight 2']
                }
            ]
        }
        mock_analyzer.analyze_standalone.return_value = mock_result
        mock_analyzer_class.return_value = mock_analyzer

        event = {
            'body': json.dumps({
                'surveyTitle': 'Test Survey',
                'theme': 'Test Theme',
                'baseline': [
                    {'sentence': 'Test sentence', 'id': '1'}
                ]
            })
        }

        response = lambda_handler(event, None)

        assert response['statusCode'] == 200
        assert 'Access-Control-Allow-Origin' in response['headers']
        body = json.loads(response['body'])
        assert 'clusters' in body
        assert len(body['clusters']) == 1

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    @patch('src.services.text_analyzer.TextAnalyzer')
    def test_lambda_handler_comparative_success(self, mock_analyzer_class):
        """Test successful comparative analysis via Lambda."""
        # Setup mock
        mock_analyzer = Mock()
        mock_result = Mock()
        mock_result.clusters = [Mock()]  # Add clusters attribute for len() call
        mock_result.model_dump.return_value = {
            'clusters': [
                {
                    'title': 'Test Cluster',
                    'sentiment': 'neutral',
                    'baselineSentences': ['1'],
                    'comparisonSentences': ['2'],
                    'keySimilarities': ['Sim 1', 'Sim 2'],
                    'keyDifferences': ['Diff 1', 'Diff 2']
                }
            ]
        }
        mock_analyzer.analyze_comparative.return_value = mock_result
        mock_analyzer_class.return_value = mock_analyzer

        event = {
            'body': json.dumps({
                'surveyTitle': 'Test Survey',
                'theme': 'Test Theme',
                'baseline': [{'sentence': 'Test 1', 'id': '1'}],
                'comparison': [{'sentence': 'Test 2', 'id': '2'}]
            })
        }

        response = lambda_handler(event, None)

        assert response['statusCode'] == 200
        body = json.loads(response['body'])
        assert 'clusters' in body

    @patch.dict('os.environ', {'OPENAI_API_KEY': ''}, clear=True)
    def test_lambda_handler_missing_api_key(self):
        """Test error when API key is missing."""
        event = {
            'body': json.dumps({
                'surveyTitle': 'Test',
                'theme': 'Theme',
                'baseline': [{'sentence': 'Test', 'id': '1'}]
            })
        }

        response = lambda_handler(event, None)

        assert response['statusCode'] == 500
        body = json.loads(response['body'])
        assert body['error']['code'] == 'CONFIGURATION_ERROR'

    def test_lambda_handler_invalid_json(self):
        """Test error handling for invalid JSON."""
        event = {
            'body': 'invalid json {'
        }

        response = lambda_handler(event, None)

        assert response['statusCode'] == 400
        body = json.loads(response['body'])
        assert body['error']['code'] == 'INVALID_JSON'

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_lambda_handler_validation_error(self):
        """Test validation error handling."""
        event = {
            'body': json.dumps({
                'surveyTitle': 'Test',
                'theme': 'Theme',
                'baseline': []  # Empty baseline - should fail validation
            })
        }

        response = lambda_handler(event, None)

        assert response['statusCode'] == 400
        body = json.loads(response['body'])
        assert body['error']['code'] == 'VALIDATION_ERROR'

    @patch.dict('os.environ', {
        'OPENAI_API_KEY': 'test-key',
        'EMBEDDING_MODEL': 'custom-embed',
        'SENTIMENT_MODEL': 'custom-sentiment',
        'MIN_CLUSTER_SIZE': '5',
        'MAX_CLUSTERS': '15'
    })
    @patch('src.services.text_analyzer.TextAnalyzer')
    def test_lambda_handler_custom_config(self, mock_analyzer_class):
        """Test that custom configuration is used."""
        mock_analyzer = Mock()
        mock_result = Mock()
        mock_result.clusters = []  # Add clusters attribute for len() call
        mock_result.model_dump.return_value = {'clusters': []}
        mock_analyzer.analyze_standalone.return_value = mock_result
        mock_analyzer_class.return_value = mock_analyzer

        event = {
            'body': json.dumps({
                'surveyTitle': 'Test',
                'theme': 'Theme',
                'baseline': [{'sentence': 'Test', 'id': '1'}]
            })
        }

        lambda_handler(event, None)

        # Verify TextAnalyzer was initialized with custom config
        mock_analyzer_class.assert_called_once_with(
            openai_api_key='test-key',
            embedding_model='custom-embed',
            sentiment_model='custom-sentiment',
            min_cluster_size=5,
            max_clusters=15
        )

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    @patch('src.services.text_analyzer.TextAnalyzer')
    def test_lambda_handler_analysis_error(self, mock_analyzer_class):
        """Test error handling during analysis."""
        mock_analyzer = Mock()
        mock_analyzer.analyze_standalone.side_effect = Exception("Analysis failed")
        mock_analyzer_class.return_value = mock_analyzer

        event = {
            'body': json.dumps({
                'surveyTitle': 'Test',
                'theme': 'Theme',
                'baseline': [{'sentence': 'Test', 'id': '1'}]
            })
        }

        response = lambda_handler(event, None)

        assert response['statusCode'] == 500
        body = json.loads(response['body'])
        assert body['error']['code'] == 'ANALYSIS_ERROR'


class TestHandleStandaloneAnalysis:
    """Test standalone analysis handler."""

    @patch('src.services.text_analyzer.TextAnalyzer')
    def test_handle_standalone_analysis_success(self, mock_analyzer_class):
        """Test successful standalone analysis."""
        mock_analyzer = Mock()
        mock_result = Mock()
        mock_result.clusters = []  # Add clusters attribute for len() call
        mock_result.model_dump.return_value = {'clusters': []}
        mock_analyzer.analyze_standalone.return_value = mock_result
        mock_analyzer_class.return_value = mock_analyzer

        body = {
            'surveyTitle': 'Test',
            'theme': 'Theme',
            'baseline': [{'sentence': 'Test', 'id': '1'}]
        }

        response = handle_standalone_analysis(
            body, mock_analyzer
        )

        assert response['statusCode'] == 200
        mock_analyzer.analyze_standalone.assert_called_once()

    @patch('src.services.text_analyzer.TextAnalyzer')
    def test_handle_standalone_analysis_value_error(self, mock_analyzer_class):
        """Test value error handling."""
        mock_analyzer = Mock()
        mock_analyzer.analyze_standalone.side_effect = ValueError("Invalid input")
        mock_analyzer_class.return_value = mock_analyzer

        body = {
            'surveyTitle': 'Test',
            'theme': 'Theme',
            'baseline': [{'sentence': 'Test', 'id': '1'}]
        }

        response = handle_standalone_analysis(
            body, mock_analyzer
        )

        assert response['statusCode'] == 400
        body_json = json.loads(response['body'])
        assert body_json['error']['code'] == 'INVALID_INPUT'


class TestHandleComparativeAnalysis:
    """Test comparative analysis handler."""

    @patch('src.services.text_analyzer.TextAnalyzer')
    def test_handle_comparative_analysis_success(self, mock_analyzer_class):
        """Test successful comparative analysis."""
        mock_analyzer = Mock()
        mock_result = Mock()
        mock_result.clusters = []  # Add clusters attribute for len() call
        mock_result.model_dump.return_value = {'clusters': []}
        mock_analyzer.analyze_comparative.return_value = mock_result
        mock_analyzer_class.return_value = mock_analyzer

        body = {
            'surveyTitle': 'Test',
            'theme': 'Theme',
            'baseline': [{'sentence': 'Test 1', 'id': '1'}],
            'comparison': [{'sentence': 'Test 2', 'id': '2'}]
        }

        response = handle_comparative_analysis(
            body, mock_analyzer
        )

        assert response['statusCode'] == 200
        mock_analyzer.analyze_comparative.assert_called_once()


class TestCreateErrorResponse:
    """Test error response creation."""

    def test_create_error_response_basic(self):
        """Test basic error response."""
        response = create_error_response(
            400, 'TEST_ERROR', 'Test error message'
        )

        assert response['statusCode'] == 400
        assert response['headers']['Content-Type'] == 'application/json'
        assert 'Access-Control-Allow-Origin' in response['headers']

        body = json.loads(response['body'])
        assert body['error']['code'] == 'TEST_ERROR'
        assert body['error']['message'] == 'Test error message'
        assert 'details' not in body['error']

    def test_create_error_response_with_details(self):
        """Test error response with details."""
        response = create_error_response(
            400,
            'VALIDATION_ERROR',
            'Validation failed',
            {'field': 'baseline', 'issue': 'Empty'}
        )

        body = json.loads(response['body'])
        assert body['error']['details']['field'] == 'baseline'
        assert body['error']['details']['issue'] == 'Empty'

    def test_create_error_response_cors_headers(self):
        """Test that CORS headers are included."""
        response = create_error_response(500, 'ERROR', 'Message')

        assert response['headers']['Access-Control-Allow-Origin'] == '*'
