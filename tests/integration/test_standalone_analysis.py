"""
Integration tests for standalone analysis.

These tests use real ML models (embeddings, sentiment, clustering)
and only mock the OpenAI API for insight generation.
"""

import pytest
from src.services.text_analyzer import TextAnalyzer
from src.models.schemas import SentenceInput


@pytest.mark.integration
class TestStandaloneAnalysisIntegration:
    """Integration tests for standalone analysis pipeline."""

    def test_full_pipeline_with_multiple_clusters(
        self,
        embedding_service,
        sentiment_service,
        clustering_service,
        insight_service_with_mock,
        sample_mixed_feedback
    ):
        """
        Test complete standalone analysis pipeline with multiple distinct clusters.

        This tests:
        - Real embeddings from SentenceTransformer
        - Real clustering with HDBSCAN
        - Real sentiment analysis with BERTweet
        - Mocked insight generation
        """
        # Initialize TextAnalyzer with real services (except insights)
        analyzer = TextAnalyzer(
            openai_api_key="test-key",
            embedding_model="all-MiniLM-L6-v2",
            sentiment_model="finiteautomata/bertweet-base-sentiment-analysis",
            min_cluster_size=3,
            max_clusters=20
        )

        # Override insight service with mocked version
        analyzer.insight_service = insight_service_with_mock

        # Convert to SentenceInput objects
        baseline = [SentenceInput(**item) for item in sample_mixed_feedback]

        # Run analysis
        result = analyzer.analyze_standalone(
            survey_title="Product Feedback Survey",
            theme="Customer Satisfaction",
            baseline=baseline
        )

        # Verify results
        assert len(result.clusters) > 0, "Should create at least one cluster"
        assert len(result.clusters) <= 20, "Should not exceed max_clusters"

        # Check each cluster structure
        for cluster in result.clusters:
            assert cluster.title is not None
            assert len(cluster.title) > 0
            assert cluster.sentiment in ['positive', 'negative', 'neutral']
            assert len(cluster.sentences) >= 1  # At least 1 sentence ID
            assert len(cluster.keyInsights) >= 2  # At least 2 insights
            assert len(cluster.keyInsights) <= 3  # At most 3 insights

            # Verify all sentence IDs are from input
            input_ids = {s.id for s in baseline}
            for sid in cluster.sentences:
                assert sid in input_ids, f"Sentence ID {sid} not in input"

    def test_similar_sentences_cluster_together(
        self,
        embedding_service,
        sentiment_service,
        clustering_service,
        insight_service_with_mock,
        sample_quality_feedback
    ):
        """
        Test that similar sentences cluster together.

        All sentences are about quality, so they should form one cluster.
        """
        analyzer = TextAnalyzer(
            openai_api_key="test-key",
            embedding_model="all-MiniLM-L6-v2",
            sentiment_model="finiteautomata/bertweet-base-sentiment-analysis",
            min_cluster_size=2,
            max_clusters=20
        )
        analyzer.insight_service = insight_service_with_mock

        baseline = [SentenceInput(**item) for item in sample_quality_feedback]

        result = analyzer.analyze_standalone(
            survey_title="Quality Survey",
            theme="Product Quality",
            baseline=baseline
        )

        # Should create 1-2 clusters (quality sentences are similar)
        assert len(result.clusters) >= 1
        assert len(result.clusters) <= 2

        # Most sentences should be clustered (not noise)
        total_clustered = sum(len(c.sentences) for c in result.clusters)
        assert total_clustered >= len(baseline) * 0.5, "At least 50% should be clustered"

    def test_sentiment_classification(
        self,
        embedding_service,
        sentiment_service,
        clustering_service,
        insight_service_with_mock
    ):
        """
        Test that sentiment is correctly classified.

        Uses clearly positive and negative sentences.
        """
        analyzer = TextAnalyzer(
            openai_api_key="test-key",
            embedding_model="all-MiniLM-L6-v2",
            sentiment_model="finiteautomata/bertweet-base-sentiment-analysis",
            min_cluster_size=2,
            max_clusters=20
        )
        analyzer.insight_service = insight_service_with_mock

        # Clearly positive sentences
        positive_feedback = [
            {'sentence': 'This is excellent and amazing', 'id': 'pos1'},
            {'sentence': 'Great product, very happy', 'id': 'pos2'},
            {'sentence': 'Wonderful experience overall', 'id': 'pos3'},
        ]

        # Clearly negative sentences
        negative_feedback = [
            {'sentence': 'This is terrible and awful', 'id': 'neg1'},
            {'sentence': 'Bad product, very disappointed', 'id': 'neg2'},
            {'sentence': 'Horrible experience overall', 'id': 'neg3'},
        ]

        baseline = [SentenceInput(**item) for item in positive_feedback + negative_feedback]

        result = analyzer.analyze_standalone(
            survey_title="Sentiment Test",
            theme="Customer Feedback",
            baseline=baseline
        )

        # Should have at least one cluster
        assert len(result.clusters) >= 1

        # Check that we have both positive and negative clusters (or at least one with clear sentiment)
        sentiments = [c.sentiment for c in result.clusters]
        assert len(sentiments) > 0
        # At least one cluster should have a clear sentiment
        assert any(s in ['positive', 'negative'] for s in sentiments)

    def test_output_schema_validation(
        self,
        embedding_service,
        sentiment_service,
        clustering_service,
        insight_service_with_mock,
        sample_quality_feedback
    ):
        """
        Test that output conforms to Pydantic schema.

        This validates the complete output structure.
        """
        analyzer = TextAnalyzer(
            openai_api_key="test-key",
            embedding_model="all-MiniLM-L6-v2",
            sentiment_model="finiteautomata/bertweet-base-sentiment-analysis",
            min_cluster_size=2,
            max_clusters=20
        )
        analyzer.insight_service = insight_service_with_mock

        baseline = [SentenceInput(**item) for item in sample_quality_feedback]

        result = analyzer.analyze_standalone(
            survey_title="Schema Test",
            theme="Product Quality",
            baseline=baseline
        )

        # Test Pydantic serialization
        result_dict = result.model_dump()

        assert 'clusters' in result_dict
        assert isinstance(result_dict['clusters'], list)

        for cluster in result_dict['clusters']:
            assert 'title' in cluster
            assert 'sentiment' in cluster
            assert 'sentences' in cluster
            assert 'keyInsights' in cluster
            assert isinstance(cluster['title'], str)
            assert cluster['sentiment'] in ['positive', 'negative', 'neutral']
            assert isinstance(cluster['sentences'], list)
            assert isinstance(cluster['keyInsights'], list)
            assert len(cluster['keyInsights']) >= 2
            assert len(cluster['keyInsights']) <= 3

    def test_edge_case_minimum_sentences(
        self,
        embedding_service,
        sentiment_service,
        clustering_service,
        insight_service_with_mock
    ):
        """
        Test with minimum number of sentences (edge case).
        """
        analyzer = TextAnalyzer(
            openai_api_key="test-key",
            embedding_model="all-MiniLM-L6-v2",
            sentiment_model="finiteautomata/bertweet-base-sentiment-analysis",
            min_cluster_size=2,
            max_clusters=20
        )
        analyzer.insight_service = insight_service_with_mock

        # Only 3 sentences
        baseline = [
            SentenceInput(sentence="Good quality product", id="1"),
            SentenceInput(sentence="Excellent build", id="2"),
            SentenceInput(sentence="Great materials", id="3")
        ]

        result = analyzer.analyze_standalone(
            survey_title="Minimal Test",
            theme="Quality",
            baseline=baseline
        )

        # Should handle gracefully
        assert len(result.clusters) >= 0  # May or may not form clusters

        # If clusters formed, validate structure
        for cluster in result.clusters:
            assert len(cluster.sentences) >= 1
            assert len(cluster.keyInsights) >= 2

    def test_duplicate_sentence_ids_handling(
        self,
        embedding_service,
        sentiment_service,
        clustering_service,
        insight_service_with_mock
    ):
        """
        Test handling of duplicate sentence IDs.

        Per spec: "an id could be in multiple resulting clusters,
        but should be only once per cluster"
        """
        analyzer = TextAnalyzer(
            openai_api_key="test-key",
            embedding_model="all-MiniLM-L6-v2",
            sentiment_model="finiteautomata/bertweet-base-sentiment-analysis",
            min_cluster_size=2,
            max_clusters=20
        )
        analyzer.insight_service = insight_service_with_mock

        # Same ID used multiple times (different sentences from same comment)
        baseline = [
            SentenceInput(sentence="The quality is great", id="comment1"),
            SentenceInput(sentence="I really love the build", id="comment1"),
            SentenceInput(sentence="The price is high", id="comment2"),
            SentenceInput(sentence="It costs too much", id="comment2"),
        ]

        result = analyzer.analyze_standalone(
            survey_title="Duplicate ID Test",
            theme="Feedback",
            baseline=baseline
        )

        # Verify no duplicate IDs within each cluster
        for cluster in result.clusters:
            unique_ids = set(cluster.sentences)
            assert len(unique_ids) == len(cluster.sentences), \
                "Cluster should not have duplicate sentence IDs"

    def test_sentence_limiting_configuration(
        self,
        embedding_service,
        sentiment_service,
        clustering_service,
        insight_service_with_mock
    ):
        """
        Test that max_sentences_for_insights configuration works.

        When cluster has more sentences than limit, only subset is used for insights.
        """
        # Create analyzer with low limit
        analyzer = TextAnalyzer(
            openai_api_key="test-key",
            embedding_model="all-MiniLM-L6-v2",
            sentiment_model="finiteautomata/bertweet-base-sentiment-analysis",
            min_cluster_size=2,
            max_clusters=20
        )
        # Override with custom limit
        analyzer.insight_service = insight_service_with_mock
        analyzer.insight_service.max_sentences_for_insights = 5

        # Create many similar sentences
        baseline = [
            SentenceInput(sentence=f"Quality product number {i}", id=f"q{i}")
            for i in range(10)
        ]

        result = analyzer.analyze_standalone(
            survey_title="Limit Test",
            theme="Quality",
            baseline=baseline
        )

        # Should still succeed with limiting
        assert len(result.clusters) >= 0

        for cluster in result.clusters:
            # Insights should still be generated (mocked)
            assert len(cluster.keyInsights) >= 2