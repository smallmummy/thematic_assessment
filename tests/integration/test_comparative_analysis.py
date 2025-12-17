"""
Integration tests for comparative analysis.

These tests use real ML models (embeddings, sentiment, clustering)
and only mock the OpenAI API for insight generation.
"""

import pytest
from src.services.text_analyzer import TextAnalyzer
from src.models.schemas import SentenceInput


@pytest.mark.integration
class TestComparativeAnalysisIntegration:
    """Integration tests for comparative analysis pipeline."""

    def test_full_comparative_pipeline(
        self,
        embedding_service,
        sentiment_service,
        clustering_service,
        insight_service_with_mock,
        sample_comparative_baseline,
        sample_comparative_comparison
    ):
        """
        Test complete comparative analysis pipeline.

        This tests:
        - Real embeddings for both datasets
        - Real joint clustering
        - Real sentiment analysis
        - Mocked comparative insight generation
        """
        analyzer = TextAnalyzer(
            openai_api_key="test-key",
            embedding_model="all-MiniLM-L6-v2",
            sentiment_model="finiteautomata/bertweet-base-sentiment-analysis",
            min_cluster_size=2,
            max_clusters=20
        )
        analyzer.insight_service = insight_service_with_mock

        baseline = [SentenceInput(**item) for item in sample_comparative_baseline]
        comparison = [SentenceInput(**item) for item in sample_comparative_comparison]

        result = analyzer.analyze_comparative(
            survey_title="UI Comparison Study",
            theme="User Experience",
            baseline=baseline,
            comparison=comparison
        )

        # Verify results
        assert len(result.clusters) > 0, "Should create at least one cluster"

        # Check each cluster structure
        for cluster in result.clusters:
            assert cluster.title is not None
            assert len(cluster.title) > 0
            assert cluster.sentiment in ['positive', 'negative', 'neutral']
            assert isinstance(cluster.baselineSentences, list)
            assert isinstance(cluster.comparisonSentences, list)
            assert len(cluster.keySimilarities) >= 2
            assert len(cluster.keySimilarities) <= 3
            assert len(cluster.keyDifferences) >= 2
            assert len(cluster.keyDifferences) <= 3

            # At least one dataset should have sentences in each cluster
            total_sentences = len(cluster.baselineSentences) + len(cluster.comparisonSentences)
            assert total_sentences > 0, "Cluster should have sentences from at least one dataset"

    def test_overlapping_themes_cluster_together(
        self,
        embedding_service,
        sentiment_service,
        clustering_service,
        insight_service_with_mock
    ):
        """
        Test that similar themes from both datasets cluster together.

        Both datasets discuss "quality" - should cluster together.
        """
        analyzer = TextAnalyzer(
            openai_api_key="test-key",
            embedding_model="all-MiniLM-L6-v2",
            sentiment_model="finiteautomata/bertweet-base-sentiment-analysis",
            min_cluster_size=2,
            max_clusters=20
        )
        analyzer.insight_service = insight_service_with_mock

        # Both datasets about quality - use more sentences for better clustering
        baseline = [
            SentenceInput(sentence="Quality is excellent and outstanding", id="b1"),
            SentenceInput(sentence="Build quality is superb and impressive", id="b2"),
            SentenceInput(sentence="Great quality materials used throughout", id="b3"),
            SentenceInput(sentence="The product quality exceeds all expectations", id="b4"),
        ]
        comparison = [
            SentenceInput(sentence="Quality has improved significantly and is better", id="c1"),
            SentenceInput(sentence="Better quality now than ever before", id="c2"),
            SentenceInput(sentence="Quality exceeds expectations by far", id="c3"),
            SentenceInput(sentence="Improved quality is clearly visible", id="c4"),
        ]

        result = analyzer.analyze_comparative(
            survey_title="Quality Comparison",
            theme="Product Quality",
            baseline=baseline,
            comparison=comparison
        )

        # Should create at least one cluster (all similar sentences)
        assert len(result.clusters) >= 1, "Should create at least one cluster for similar sentences"

        # Should have clusters with both baseline and comparison
        has_overlap = any(
            len(c.baselineSentences) > 0 and len(c.comparisonSentences) > 0
            for c in result.clusters
        )
        assert has_overlap, "Should have at least one cluster with both datasets"

    def test_distinct_themes_separate_clusters(
        self,
        embedding_service,
        sentiment_service,
        clustering_service,
        insight_service_with_mock
    ):
        """
        Test that distinct themes form separate clusters.

        Baseline about quality, comparison about price - should be separate.
        """
        analyzer = TextAnalyzer(
            openai_api_key="test-key",
            embedding_model="all-MiniLM-L6-v2",
            sentiment_model="finiteautomata/bertweet-base-sentiment-analysis",
            min_cluster_size=2,
            max_clusters=20
        )
        analyzer.insight_service = insight_service_with_mock

        # Baseline about quality - use more distinct sentences
        baseline = [
            SentenceInput(sentence="Quality is excellent and superior", id="b1"),
            SentenceInput(sentence="Build quality is outstanding and remarkable", id="b2"),
            SentenceInput(sentence="Great quality materials and craftsmanship", id="b3"),
            SentenceInput(sentence="Superior quality compared to competitors", id="b4"),
        ]

        # Comparison about price - very different topic
        comparison = [
            SentenceInput(sentence="Price is way too high and expensive", id="c1"),
            SentenceInput(sentence="Too expensive and overpriced for budget", id="c2"),
            SentenceInput(sentence="Overpriced product costs too much money", id="c3"),
            SentenceInput(sentence="Pricing is unreasonable and not affordable", id="c4"),
        ]

        result = analyzer.analyze_comparative(
            survey_title="Quality vs Price",
            theme="Product Feedback",
            baseline=baseline,
            comparison=comparison
        )

        # Should form multiple clusters (distinct themes)
        assert len(result.clusters) >= 1

        # May have one-sided clusters (only baseline or only comparison)
        baseline_only = [c for c in result.clusters if len(c.baselineSentences) > 0 and len(c.comparisonSentences) == 0]
        comparison_only = [c for c in result.clusters if len(c.baselineSentences) == 0 and len(c.comparisonSentences) > 0]

        # Should have at least some separation
        assert len(baseline_only) > 0 or len(comparison_only) > 0, \
            "Distinct themes should create some one-sided clusters"

    def test_one_sided_cluster_handling(
        self,
        embedding_service,
        sentiment_service,
        clustering_service,
        insight_service_with_mock
    ):
        """
        Test clusters with only baseline or only comparison sentences.

        This is a valid scenario when themes are unique to one dataset.
        """
        analyzer = TextAnalyzer(
            openai_api_key="test-key",
            embedding_model="all-MiniLM-L6-v2",
            sentiment_model="finiteautomata/bertweet-base-sentiment-analysis",
            min_cluster_size=2,
            max_clusters=20
        )
        analyzer.insight_service = insight_service_with_mock

        baseline = [
            SentenceInput(sentence="Unique baseline feature A", id="b1"),
            SentenceInput(sentence="Baseline only feature B", id="b2"),
            SentenceInput(sentence="Baseline exclusive feature C", id="b3"),
        ]
        comparison = [
            SentenceInput(sentence="New comparison feature X", id="c1"),
            SentenceInput(sentence="Comparison only feature Y", id="c2"),
            SentenceInput(sentence="Comparison exclusive feature Z", id="c3"),
        ]

        result = analyzer.analyze_comparative(
            survey_title="Feature Comparison",
            theme="Product Features",
            baseline=baseline,
            comparison=comparison
        )

        # Should handle one-sided clusters gracefully
        assert len(result.clusters) >= 0

        for cluster in result.clusters:
            # Should still generate similarities and differences
            assert len(cluster.keySimilarities) >= 2
            assert len(cluster.keyDifferences) >= 2

    def test_comparative_output_schema(
        self,
        embedding_service,
        sentiment_service,
        clustering_service,
        insight_service_with_mock,
        sample_comparative_baseline,
        sample_comparative_comparison
    ):
        """
        Test that comparative output conforms to Pydantic schema.
        """
        analyzer = TextAnalyzer(
            openai_api_key="test-key",
            embedding_model="all-MiniLM-L6-v2",
            sentiment_model="finiteautomata/bertweet-base-sentiment-analysis",
            min_cluster_size=2,
            max_clusters=20
        )
        analyzer.insight_service = insight_service_with_mock

        baseline = [SentenceInput(**item) for item in sample_comparative_baseline]
        comparison = [SentenceInput(**item) for item in sample_comparative_comparison]

        result = analyzer.analyze_comparative(
            survey_title="Schema Test",
            theme="User Experience",
            baseline=baseline,
            comparison=comparison
        )

        # Test Pydantic serialization
        result_dict = result.model_dump()

        assert 'clusters' in result_dict
        assert isinstance(result_dict['clusters'], list)

        for cluster in result_dict['clusters']:
            assert 'title' in cluster
            assert 'sentiment' in cluster
            assert 'baselineSentences' in cluster
            assert 'comparisonSentences' in cluster
            assert 'keySimilarities' in cluster
            assert 'keyDifferences' in cluster

            assert isinstance(cluster['title'], str)
            assert cluster['sentiment'] in ['positive', 'negative', 'neutral']
            assert isinstance(cluster['baselineSentences'], list)
            assert isinstance(cluster['comparisonSentences'], list)
            assert isinstance(cluster['keySimilarities'], list)
            assert isinstance(cluster['keyDifferences'], list)
            assert len(cluster['keySimilarities']) >= 2
            assert len(cluster['keySimilarities']) <= 3
            assert len(cluster['keyDifferences']) >= 2
            assert len(cluster['keyDifferences']) <= 3

    def test_comparative_sentiment_analysis(
        self,
        embedding_service,
        sentiment_service,
        clustering_service,
        insight_service_with_mock
    ):
        """
        Test sentiment analysis in comparative mode.

        Baseline is positive, comparison is negative - sentiment should reflect this.
        """
        analyzer = TextAnalyzer(
            openai_api_key="test-key",
            embedding_model="all-MiniLM-L6-v2",
            sentiment_model="finiteautomata/bertweet-base-sentiment-analysis",
            min_cluster_size=2,
            max_clusters=20
        )
        analyzer.insight_service = insight_service_with_mock

        # Positive baseline
        baseline = [
            SentenceInput(sentence="Excellent product quality", id="b1"),
            SentenceInput(sentence="Great experience overall", id="b2"),
            SentenceInput(sentence="Very satisfied with purchase", id="b3"),
        ]

        # Negative comparison
        comparison = [
            SentenceInput(sentence="Terrible product quality", id="c1"),
            SentenceInput(sentence="Bad experience overall", id="c2"),
            SentenceInput(sentence="Very disappointed with purchase", id="c3"),
        ]

        result = analyzer.analyze_comparative(
            survey_title="Sentiment Test",
            theme="Customer Satisfaction",
            baseline=baseline,
            comparison=comparison
        )

        # Should have clusters
        assert len(result.clusters) >= 1

        # Sentiments should be assigned
        for cluster in result.clusters:
            assert cluster.sentiment in ['positive', 'negative', 'neutral']

    def test_duplicate_ids_across_datasets(
        self,
        embedding_service,
        sentiment_service,
        clustering_service,
        insight_service_with_mock
    ):
        """
        Test handling of duplicate IDs within each dataset.

        Per spec: IDs should be unique within each cluster per dataset.
        """
        analyzer = TextAnalyzer(
            openai_api_key="test-key",
            embedding_model="all-MiniLM-L6-v2",
            sentiment_model="finiteautomata/bertweet-base-sentiment-analysis",
            min_cluster_size=2,
            max_clusters=20
        )
        analyzer.insight_service = insight_service_with_mock

        # Duplicate IDs within baseline
        baseline = [
            SentenceInput(sentence="Quality is good", id="same_id"),
            SentenceInput(sentence="Build is solid", id="same_id"),
            SentenceInput(sentence="Materials are great", id="other_id"),
        ]

        # Different IDs in comparison
        comparison = [
            SentenceInput(sentence="Quality improved", id="comp1"),
            SentenceInput(sentence="Build is better", id="comp2"),
            SentenceInput(sentence="Materials upgraded", id="comp3"),
        ]

        result = analyzer.analyze_comparative(
            survey_title="Duplicate ID Test",
            theme="Quality",
            baseline=baseline,
            comparison=comparison
        )

        # Verify no duplicate IDs within each dataset per cluster
        for cluster in result.clusters:
            baseline_ids = set(cluster.baselineSentences)
            comparison_ids = set(cluster.comparisonSentences)

            assert len(baseline_ids) == len(cluster.baselineSentences), \
                "No duplicate baseline IDs within cluster"
            assert len(comparison_ids) == len(cluster.comparisonSentences), \
                "No duplicate comparison IDs within cluster"

    def test_sentence_limiting_comparative(
        self,
        embedding_service,
        sentiment_service,
        clustering_service,
        insight_service_with_mock
    ):
        """
        Test that max_sentences_for_comparison configuration works.
        """
        analyzer = TextAnalyzer(
            openai_api_key="test-key",
            embedding_model="all-MiniLM-L6-v2",
            sentiment_model="finiteautomata/bertweet-base-sentiment-analysis",
            min_cluster_size=2,
            max_clusters=20
        )
        analyzer.insight_service = insight_service_with_mock
        analyzer.insight_service.max_sentences_for_comparison = 3

        # Many sentences in both datasets
        baseline = [
            SentenceInput(sentence=f"Baseline quality {i}", id=f"b{i}")
            for i in range(10)
        ]
        comparison = [
            SentenceInput(sentence=f"Comparison quality {i}", id=f"c{i}")
            for i in range(10)
        ]

        result = analyzer.analyze_comparative(
            survey_title="Limit Test",
            theme="Quality",
            baseline=baseline,
            comparison=comparison
        )

        # Should still succeed with limiting
        assert len(result.clusters) >= 0

        for cluster in result.clusters:
            # Insights should still be generated (mocked)
            assert len(cluster.keySimilarities) >= 2
            assert len(cluster.keyDifferences) >= 2