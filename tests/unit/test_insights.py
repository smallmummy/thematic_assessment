"""
Unit tests for InsightService.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from src.services.insights import InsightService


class TestInsightService:
    """Test InsightService class."""

    @patch('src.services.insights.AsyncOpenAI')
    def test_initialization(self, mock_async_openai):
        """Test service initialization."""
        service = InsightService(
            api_key="test-key",
            model="gpt-4o",
            max_sentences_for_insights=15,
            max_sentences_for_comparison=10,
            max_title_length=80
        )

        mock_async_openai.assert_called_once_with(api_key="test-key")
        assert service.model == "gpt-4o"
        assert service.max_sentences_for_insights == 15
        assert service.max_sentences_for_comparison == 10
        assert service.max_title_length == 80

    @patch('src.services.insights.AsyncOpenAI')
    def test_initialization_defaults(self, mock_async_openai):
        """Test service initialization with defaults."""
        service = InsightService(api_key="test-key")

        assert service.model == "gpt-4o"
        assert service.max_sentences_for_insights == 20
        assert service.max_sentences_for_comparison == 15
        assert service.max_title_length == 100

    @patch('src.services.insights.AsyncOpenAI')
    def test_generate_standalone_cluster_content_async_success(self, mock_async_openai):
        """Test successful standalone content generation."""
        # Setup mock
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''
        {
            "title": "Test Title",
            "insights": [
                "Insight **one** is here",
                "Insight **two** is here"
            ]
        }
        '''

        mock_client = Mock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_async_openai.return_value = mock_client

        service = InsightService(api_key="test-key")
        sentences = ["Test sentence 1", "Test sentence 2"]

        result = asyncio.run(service.generate_standalone_cluster_content_async(
            sentences, "Test Theme", "positive"
        ))

        assert result['title'] == "Test Title"
        assert len(result['insights']) == 2
        assert "Insight **one**" in result['insights'][0]

    @patch('src.services.insights.AsyncOpenAI')
    def test_generate_standalone_cluster_content_async_sentence_limiting(
        self, mock_async_openai
    ):
        """Test that sentences are limited to max_sentences_for_insights."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''
        {
            "title": "Test",
            "insights": ["Insight 1", "Insight 2"]
        }
        '''

        mock_client = Mock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_async_openai.return_value = mock_client

        service = InsightService(api_key="test-key", max_sentences_for_insights=5)
        sentences = [f"Sentence {i}" for i in range(10)]  # 10 sentences

        asyncio.run(service.generate_standalone_cluster_content_async(
            sentences, "Theme", "positive"
        ))

        # Check that only 5 sentences were passed in prompt
        call_args = mock_client.chat.completions.create.call_args
        prompt = call_args[1]['messages'][1]['content']
        # Count sentence markers in prompt
        sentence_count = prompt.count("- Sentence")
        assert sentence_count == 5

    @patch('src.services.insights.AsyncOpenAI')
    def test_generate_standalone_cluster_content_async_no_limiting_when_under_max(
        self, mock_async_openai
    ):
        """Test that sentences are not limited when under max."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''
        {
            "title": "Test",
            "insights": ["Insight 1", "Insight 2"]
        }
        '''

        mock_client = Mock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_async_openai.return_value = mock_client

        service = InsightService(api_key="test-key", max_sentences_for_insights=10)
        sentences = [f"Sentence {i}" for i in range(5)]  # 5 sentences, under limit

        asyncio.run(service.generate_standalone_cluster_content_async(
            sentences, "Theme", "positive"
        ))

        # Check that all 5 sentences were passed
        call_args = mock_client.chat.completions.create.call_args
        prompt = call_args[1]['messages'][1]['content']
        sentence_count = prompt.count("- Sentence")
        assert sentence_count == 5

    @patch('src.services.insights.AsyncOpenAI')
    def test_generate_standalone_cluster_content_async_title_truncation(
        self, mock_async_openai
    ):
        """Test title truncation when too long."""
        long_title = "A" * 150  # 150 characters
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = f'''
        {{
            "title": "{long_title}",
            "insights": ["Insight 1", "Insight 2"]
        }}
        '''

        mock_client = Mock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_async_openai.return_value = mock_client

        service = InsightService(api_key="test-key", max_title_length=100)

        result = asyncio.run(service.generate_standalone_cluster_content_async(
            ["Test"], "Theme", "positive"
        ))

        # Title should be truncated to 100 chars with "..."
        assert len(result['title']) == 100
        assert result['title'].endswith("...")

    @patch('src.services.insights.AsyncOpenAI')
    def test_generate_standalone_cluster_content_async_insights_validation(
        self, mock_async_openai
    ):
        """Test insights count validation."""
        # Test with only 1 insight (should add fallback)
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''
        {
            "title": "Test",
            "insights": ["Only one insight"]
        }
        '''

        mock_client = Mock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_async_openai.return_value = mock_client

        service = InsightService(api_key="test-key")

        result = asyncio.run(service.generate_standalone_cluster_content_async(
            ["Test"], "Theme", "positive"
        ))

        # Should have 2 insights (1 from response + 1 fallback)
        assert len(result['insights']) == 2

    @patch('src.services.insights.AsyncOpenAI')
    def test_generate_standalone_cluster_content_async_error_fallback(
        self, mock_async_openai
    ):
        """Test fallback when generation fails."""
        mock_client = Mock()
        mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API error"))
        mock_async_openai.return_value = mock_client

        service = InsightService(api_key="test-key")

        result = asyncio.run(service.generate_standalone_cluster_content_async(
            ["Test"], "Product Quality", "positive"
        ))

        # Should return fallback content
        assert result['title'] == "Product Quality Theme"
        assert len(result['insights']) == 2
        assert "positive" in result['insights'][0]

    @patch('src.services.insights.AsyncOpenAI')
    def test_generate_comparative_cluster_content_async_success(
        self, mock_async_openai
    ):
        """Test successful comparative content generation."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''
        {
            "title": "Comparative Title",
            "similarities": ["Sim **one**", "Sim **two**"],
            "differences": ["Diff **one**", "Diff **two**"]
        }
        '''

        mock_client = Mock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_async_openai.return_value = mock_client

        service = InsightService(api_key="test-key")

        result = asyncio.run(service.generate_comparative_cluster_content_async(
            ["Baseline 1"], ["Comparison 1"], "Theme", "neutral"
        ))

        assert result['title'] == "Comparative Title"
        assert len(result['similarities']) == 2
        assert len(result['differences']) == 2

    @patch('src.services.insights.AsyncOpenAI')
    def test_generate_comparative_cluster_content_async_sentence_limiting(
        self, mock_async_openai
    ):
        """Test sentence limiting for comparative analysis."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''
        {
            "title": "Test",
            "similarities": ["Sim 1", "Sim 2"],
            "differences": ["Diff 1", "Diff 2"]
        }
        '''

        mock_client = Mock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_async_openai.return_value = mock_client

        service = InsightService(api_key="test-key", max_sentences_for_comparison=3)

        baseline = [f"Baseline {i}" for i in range(10)]
        comparison = [f"Comparison {i}" for i in range(10)]

        asyncio.run(service.generate_comparative_cluster_content_async(
            baseline, comparison, "Theme", "positive"
        ))

        # Check that sentences were limited to 3 each
        call_args = mock_client.chat.completions.create.call_args
        prompt = call_args[1]['messages'][1]['content']
        assert prompt.count("- Baseline") == 3
        assert prompt.count("- Comparison") == 3
