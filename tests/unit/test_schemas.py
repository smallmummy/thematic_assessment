"""
Unit tests for Pydantic schema models.
"""

import pytest
from pydantic import ValidationError

from src.models.schemas import (
    SentenceInput,
    StandaloneAnalysisRequest,
    ComparativeAnalysisRequest,
    StandaloneCluster,
    ComparativeCluster,
    StandaloneAnalysisResponse,
    ComparativeAnalysisResponse,
    ErrorResponse
)


class TestSentenceInput:
    """Test SentenceInput model."""

    def test_valid_sentence_input(self):
        """Test valid sentence input."""
        sentence = SentenceInput(sentence="This is a test", id="123")
        assert sentence.sentence == "This is a test"
        assert sentence.id == "123"

    def test_sentence_strips_whitespace(self):
        """Test that sentence whitespace is stripped."""
        sentence = SentenceInput(sentence="  This is a test  ", id="123")
        assert sentence.sentence == "This is a test"

    def test_id_strips_whitespace(self):
        """Test that id whitespace is stripped."""
        sentence = SentenceInput(sentence="Test", id="  123  ")
        assert sentence.id == "123"

    def test_empty_sentence_raises_error(self):
        """Test that empty sentence raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            SentenceInput(sentence="", id="123")
        assert "String should have at least 1 character" in str(exc_info.value)

    def test_whitespace_only_sentence_raises_error(self):
        """Test that whitespace-only sentence raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            SentenceInput(sentence="   ", id="123")
        assert "Sentence cannot be empty or whitespace only" in str(exc_info.value)

    def test_empty_id_raises_error(self):
        """Test that empty id raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            SentenceInput(sentence="Test", id="")
        assert "String should have at least 1 character" in str(exc_info.value)

    def test_whitespace_only_id_raises_error(self):
        """Test that whitespace-only id raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            SentenceInput(sentence="Test", id="   ")
        assert "ID cannot be empty or whitespace only" in str(exc_info.value)


class TestStandaloneAnalysisRequest:
    """Test StandaloneAnalysisRequest model."""

    def test_valid_standalone_request(self):
        """Test valid standalone analysis request."""
        request = StandaloneAnalysisRequest(
            surveyTitle="Test Survey",
            theme="Test Theme",
            baseline=[
                SentenceInput(sentence="Test 1", id="1"),
                SentenceInput(sentence="Test 2", id="2")
            ]
        )
        assert request.surveyTitle == "Test Survey"
        assert request.theme == "Test Theme"
        assert len(request.baseline) == 2

    def test_empty_baseline_raises_error(self):
        """Test that empty baseline raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            StandaloneAnalysisRequest(
                surveyTitle="Test",
                theme="Theme",
                baseline=[]
            )
        assert "List should have at least 1 item" in str(exc_info.value)

    def test_too_many_sentences_raises_error(self):
        """Test that too many sentences raises validation error."""
        baseline = [SentenceInput(sentence=f"Test {i}", id=str(i)) for i in range(10001)]
        with pytest.raises(ValidationError) as exc_info:
            StandaloneAnalysisRequest(
                surveyTitle="Test",
                theme="Theme",
                baseline=baseline
            )
        assert "List should have at most 10000 items" in str(exc_info.value)


class TestComparativeAnalysisRequest:
    """Test ComparativeAnalysisRequest model."""

    def test_valid_comparative_request(self):
        """Test valid comparative analysis request."""
        request = ComparativeAnalysisRequest(
            surveyTitle="Test Survey",
            theme="Test Theme",
            baseline=[SentenceInput(sentence="Test 1", id="1")],
            comparison=[SentenceInput(sentence="Test 2", id="2")]
        )
        assert request.surveyTitle == "Test Survey"
        assert request.theme == "Test Theme"
        assert len(request.baseline) == 1
        assert len(request.comparison) == 1

    def test_empty_baseline_raises_error(self):
        """Test that empty baseline raises validation error."""
        with pytest.raises(ValidationError):
            ComparativeAnalysisRequest(
                surveyTitle="Test",
                theme="Theme",
                baseline=[],
                comparison=[SentenceInput(sentence="Test", id="1")]
            )

    def test_empty_comparison_raises_error(self):
        """Test that empty comparison raises validation error."""
        with pytest.raises(ValidationError):
            ComparativeAnalysisRequest(
                surveyTitle="Test",
                theme="Theme",
                baseline=[SentenceInput(sentence="Test", id="1")],
                comparison=[]
            )


class TestStandaloneCluster:
    """Test StandaloneCluster model."""

    def test_valid_standalone_cluster(self):
        """Test valid standalone cluster."""
        cluster = StandaloneCluster(
            title="Test Cluster",
            sentiment="positive",
            sentences=["1", "2", "3"],
            keyInsights=["Insight 1", "Insight 2"]
        )
        assert cluster.title == "Test Cluster"
        assert cluster.sentiment == "positive"
        assert len(cluster.sentences) == 3
        assert len(cluster.keyInsights) == 2

    def test_invalid_sentiment_raises_error(self):
        """Test that invalid sentiment raises validation error."""
        with pytest.raises(ValidationError):
            StandaloneCluster(
                title="Test",
                sentiment="invalid",
                sentences=["1"],
                keyInsights=["Insight 1", "Insight 2"]
            )

    def test_too_few_insights_raises_error(self):
        """Test that too few insights raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            StandaloneCluster(
                title="Test",
                sentiment="positive",
                sentences=["1"],
                keyInsights=["Insight 1"]
            )
        assert "List should have at least 2 items" in str(exc_info.value)

    def test_too_many_insights_raises_error(self):
        """Test that too many insights raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            StandaloneCluster(
                title="Test",
                sentiment="positive",
                sentences=["1"],
                keyInsights=["Insight 1", "Insight 2", "Insight 3", "Insight 4"]
            )
        assert "List should have at most 3 items" in str(exc_info.value)


class TestComparativeCluster:
    """Test ComparativeCluster model."""

    def test_valid_comparative_cluster(self):
        """Test valid comparative cluster."""
        cluster = ComparativeCluster(
            title="Test Cluster",
            sentiment="neutral",
            baselineSentences=["1", "2"],
            comparisonSentences=["3", "4"],
            keySimilarities=["Similarity 1", "Similarity 2"],
            keyDifferences=["Difference 1", "Difference 2"]
        )
        assert cluster.title == "Test Cluster"
        assert cluster.sentiment == "neutral"
        assert len(cluster.baselineSentences) == 2
        assert len(cluster.comparisonSentences) == 2
        assert len(cluster.keySimilarities) == 2
        assert len(cluster.keyDifferences) == 2

    def test_empty_baseline_sentences_allowed(self):
        """Test that empty baseline sentences is allowed."""
        cluster = ComparativeCluster(
            title="Test",
            sentiment="positive",
            baselineSentences=[],
            comparisonSentences=["1"],
            keySimilarities=["Sim 1", "Sim 2"],
            keyDifferences=["Diff 1", "Diff 2"]
        )
        assert len(cluster.baselineSentences) == 0

    def test_too_few_similarities_raises_error(self):
        """Test that too few similarities raises validation error."""
        with pytest.raises(ValidationError):
            ComparativeCluster(
                title="Test",
                sentiment="positive",
                baselineSentences=["1"],
                comparisonSentences=["2"],
                keySimilarities=["Similarity 1"],
                keyDifferences=["Diff 1", "Diff 2"]
            )

    def test_too_many_differences_raises_error(self):
        """Test that too many differences raises validation error."""
        with pytest.raises(ValidationError):
            ComparativeCluster(
                title="Test",
                sentiment="positive",
                baselineSentences=["1"],
                comparisonSentences=["2"],
                keySimilarities=["Sim 1", "Sim 2"],
                keyDifferences=["D1", "D2", "D3", "D4"]
            )


class TestAnalysisResponses:
    """Test analysis response models."""

    def test_standalone_analysis_response(self):
        """Test standalone analysis response."""
        clusters = [
            StandaloneCluster(
                title="Cluster 1",
                sentiment="positive",
                sentences=["1", "2"],
                keyInsights=["Insight 1", "Insight 2"]
            )
        ]
        response = StandaloneAnalysisResponse(clusters=clusters)
        assert len(response.clusters) == 1
        assert response.clusters[0].title == "Cluster 1"

    def test_comparative_analysis_response(self):
        """Test comparative analysis response."""
        clusters = [
            ComparativeCluster(
                title="Cluster 1",
                sentiment="negative",
                baselineSentences=["1"],
                comparisonSentences=["2"],
                keySimilarities=["Sim 1", "Sim 2"],
                keyDifferences=["Diff 1", "Diff 2"]
            )
        ]
        response = ComparativeAnalysisResponse(clusters=clusters)
        assert len(response.clusters) == 1
        assert response.clusters[0].sentiment == "negative"


class TestErrorResponse:
    """Test ErrorResponse model."""

    def test_error_response_creation(self):
        """Test error response creation."""
        error = ErrorResponse.create(
            code="TEST_ERROR",
            message="This is a test error"
        )
        assert error.error["code"] == "TEST_ERROR"
        assert error.error["message"] == "This is a test error"
        assert "details" not in error.error

    def test_error_response_with_details(self):
        """Test error response with details."""
        error = ErrorResponse.create(
            code="VALIDATION_ERROR",
            message="Validation failed",
            details={"field": "baseline", "issue": "Empty list"}
        )
        assert error.error["code"] == "VALIDATION_ERROR"
        assert error.error["details"]["field"] == "baseline"
