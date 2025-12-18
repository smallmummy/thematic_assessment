"""
Main text analysis orchestrator that coordinates all services.
"""

import logging
import asyncio
from typing import List
import numpy as np

from src.models.schemas import (
    SentenceInput,
    StandaloneCluster,
    ComparativeCluster,
    StandaloneAnalysisResponse,
    ComparativeAnalysisResponse
)
from src.services.embeddings import EmbeddingService
from src.services.clustering import ClusteringService
from src.services.sentiment import SentimentService
from src.services.insights import InsightService
from src.utils.performance import PerformanceTimer, PerformanceTracker


logger = logging.getLogger(__name__)


class TextAnalyzer:
    """Main orchestrator for text analysis pipeline."""

    def __init__(
        self,
        openai_api_key: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english",
        min_cluster_size: int = 3,
        max_clusters: int = 20,
        clustering_config: dict = None
    ):
        """
        Initialize the text analyzer with all required services.

        Args:
            openai_api_key: API key for OpenAI
            embedding_model: Model name for embeddings
            sentiment_model: Model name for sentiment analysis
            min_cluster_size: Minimum sentences per cluster
            max_clusters: Maximum number of clusters to return
            clustering_config: Optional dict with additional clustering parameters
                (e.g., {'similarity_threshold': 0.3, 'min_samples': 1})
        """
        logger.info("Initializing TextAnalyzer")

        self.max_clusters = max_clusters

        # Prepare clustering service parameters
        clustering_params = {'min_cluster_size': min_cluster_size}
        if clustering_config:
            clustering_params.update(clustering_config)

        # Initialize services
        self.embedding_service = EmbeddingService(model_name=embedding_model)
        self.clustering_service = ClusteringService(**clustering_params)
        self.sentiment_service = SentimentService(model_name=sentiment_model)
        self.insight_service = InsightService(api_key=openai_api_key)

        logger.info("TextAnalyzer initialized successfully")

    async def _generate_standalone_insights_parallel(
        self,
        organized_clusters: dict,
        cluster_sentiments: dict,
        theme: str
    ) -> List[StandaloneCluster]:
        """
        Generate insights for all clusters in parallel using async.

        Args:
            organized_clusters: Dictionary of cluster data
            cluster_sentiments: Dictionary of cluster sentiments
            theme: Overall theme

        Returns:
            List of StandaloneCluster objects
        """
        async def process_cluster(cluster_id, cluster_data):
            """Process a single cluster."""
            try:
                # Generate title + insights in one combined call
                content = await self.insight_service.generate_standalone_cluster_content_async(
                    cluster_data['sentences'],
                    theme,
                    cluster_sentiments[cluster_id]
                )

                # Create cluster object
                cluster = StandaloneCluster(
                    title=content['title'],
                    sentiment=cluster_sentiments[cluster_id],
                    sentences=cluster_data['sentence_ids'],
                    keyInsights=content['insights']
                )

                logger.info(
                    "Processed cluster %s: '%s' "
                    "(%s unique IDs, %s)",
                    cluster_id, content['title'],
                    len(cluster_data['sentence_ids']), cluster_sentiments[cluster_id]
                )

                return cluster

            except Exception as e:
                logger.error(
                    "Error processing cluster %s: %s",
                    cluster_id, str(e)
                )
                return None

        # Process all clusters in parallel
        tasks = [
            process_cluster(cluster_id, cluster_data)
            for cluster_id, cluster_data in organized_clusters.items()
        ]

        results = await asyncio.gather(*tasks)

        # Filter out None results (failed clusters)
        return [cluster for cluster in results if cluster is not None]

    async def _generate_comparative_insights_parallel(
        self,
        organized_clusters: dict,
        cluster_sentiments: dict,
        theme: str
    ) -> List[ComparativeCluster]:
        """
        Generate comparative insights for all clusters in parallel using async.

        Args:
            organized_clusters: Dictionary of cluster data
            cluster_sentiments: Dictionary of cluster sentiments
            theme: Overall theme

        Returns:
            List of ComparativeCluster objects
        """
        async def process_cluster(cluster_id, cluster_data):
            """Process a single cluster."""
            try:
                # Skip empty clusters
                all_sentences = (
                    cluster_data['baseline_sentences'] +
                    cluster_data['comparison_sentences']
                )

                if not all_sentences:
                    return None

                # Generate title + similarities + differences in one combined call
                content = await self.insight_service.generate_comparative_cluster_content_async(
                    cluster_data['baseline_sentences'],
                    cluster_data['comparison_sentences'],
                    theme,
                    cluster_sentiments[cluster_id]
                )

                # Create cluster object
                cluster = ComparativeCluster(
                    title=content['title'],
                    sentiment=cluster_sentiments[cluster_id],
                    baselineSentences=cluster_data['baseline_sentence_ids'],
                    comparisonSentences=cluster_data['comparison_sentence_ids'],
                    keySimilarities=content['similarities'],
                    keyDifferences=content['differences']
                )

                logger.info(
                    "Processed comparative cluster %s: '%s' "
                    "(baseline: %s, comparison: %s, %s)",
                    cluster_id, content['title'],
                    len(cluster_data['baseline_sentence_ids']),
                    len(cluster_data['comparison_sentence_ids']), cluster_sentiments[cluster_id]
                )

                return cluster

            except Exception as e:
                logger.error(
                    "Error processing comparative cluster %s: %s",
                    cluster_id, str(e)
                )
                return None

        # Process all clusters in parallel
        tasks = [
            process_cluster(cluster_id, cluster_data)
            for cluster_id, cluster_data in organized_clusters.items()
        ]

        results = await asyncio.gather(*tasks)

        # Filter out None results (failed clusters)
        return [cluster for cluster in results if cluster is not None]

    async def _analyze_sentiments_parallel(
        self,
        organized_clusters: dict
    ) -> dict:
        """
        Analyze sentiment for all standalone clusters in parallel.

        Args:
            organized_clusters: Dictionary of cluster data

        Returns:
            Dictionary mapping cluster_id to sentiment label
        """
        async def analyze_cluster_sentiment(cluster_id, cluster_data):
            """Analyze sentiment for a single cluster in a thread."""
            try:
                sentiment = await asyncio.to_thread(
                    self.sentiment_service.aggregate_cluster_sentiment,
                    cluster_data['sentences']
                )
                logger.info(
                    "Analyzed sentiment for cluster %s: %s (%s sentences)",
                    cluster_id, sentiment, len(cluster_data['sentences'])
                )
                return cluster_id, sentiment
            except Exception as e:
                logger.error(
                    "Error analyzing sentiment for cluster %s: %s",
                    cluster_id, str(e)
                )
                return cluster_id, 'neutral'

        # Process all clusters in parallel
        tasks = [
            analyze_cluster_sentiment(cluster_id, cluster_data)
            for cluster_id, cluster_data in organized_clusters.items()
        ]

        results = await asyncio.gather(*tasks)

        # Convert list of tuples to dictionary
        return {cluster_id: sentiment for cluster_id, sentiment in results}

    async def _analyze_sentiments_parallel_comparative(
        self,
        organized_clusters: dict
    ) -> dict:
        """
        Analyze sentiment for all comparative clusters in parallel.

        Args:
            organized_clusters: Dictionary of cluster data with baseline/comparison

        Returns:
            Dictionary mapping cluster_id to sentiment label
        """
        async def analyze_cluster_sentiment(cluster_id, cluster_data):
            """Analyze sentiment for a single comparative cluster in a thread."""
            try:
                # Combine all sentences for sentiment analysis
                all_sentences = (
                    cluster_data['baseline_sentences'] +
                    cluster_data['comparison_sentences']
                )

                if all_sentences:
                    sentiment = await asyncio.to_thread(
                        self.sentiment_service.aggregate_cluster_sentiment,
                        all_sentences
                    )
                    logger.info(
                        "Analyzed sentiment for comparative cluster %s: %s "
                        "(baseline: %s, comparison: %s)",
                        cluster_id, sentiment,
                        len(cluster_data['baseline_sentences']),
                        len(cluster_data['comparison_sentences'])
                    )
                    return cluster_id, sentiment
                else:
                    logger.warning(
                        "Empty cluster %s, defaulting to neutral sentiment",
                        cluster_id
                    )
                    return cluster_id, 'neutral'

            except Exception as e:
                logger.error(
                    "Error analyzing sentiment for comparative cluster %s: %s",
                    cluster_id, str(e)
                )
                return cluster_id, 'neutral'

        # Process all clusters in parallel
        tasks = [
            analyze_cluster_sentiment(cluster_id, cluster_data)
            for cluster_id, cluster_data in organized_clusters.items()
        ]

        results = await asyncio.gather(*tasks)

        # Convert list of tuples to dictionary
        return {cluster_id: sentiment for cluster_id, sentiment in results}

    def analyze_standalone(
        self,
        survey_title: str,
        theme: str,
        baseline: List[SentenceInput]
    ) -> StandaloneAnalysisResponse:
        """
        Perform standalone analysis on a single dataset.

        Args:
            survey_title: Title of the survey/dataset
            theme: Overall theme of the analysis
            baseline: List of sentence inputs

        Returns:
            StandaloneAnalysisResponse with clusters
        """
        logger.info(
            "Starting standalone analysis: survey='%s', "
            "theme='%s', sentences=%s",
            survey_title, theme, len(baseline)
        )

        # Initialize performance tracker
        perf_tracker = PerformanceTracker("Standalone Analysis")

        # Step 1: Prepare sentence data
        with PerformanceTimer("1. Data Preparation") as timer:
            sentence_data = [
                {'sentence': item.sentence, 'id': item.id}
                for item in baseline
            ]
        perf_tracker.record_stage("1. Data Preparation", timer.duration)

        # Step 2: Generate embeddings
        logger.info("Step 1/4: Generating embeddings")
        with PerformanceTimer("2. Embedding Generation") as timer:
            embedding_data = self.embedding_service.generate_embeddings_with_metadata(
                sentence_data
            )

            # Convert to arrays for clustering
            embeddings = np.array([data['embedding'] for data in embedding_data.values()])
            sentence_list = [data for data in embedding_data.values()]
        perf_tracker.record_stage("2. Embedding Generation", timer.duration)

        # Step 3: Perform clustering
        logger.info("Step 2/4: Clustering sentences")
        with PerformanceTimer("3. Clustering") as timer:
            cluster_labels = self.clustering_service.cluster_embeddings(
                embeddings
            )

            # Organize clusters
            organized_clusters = self.clustering_service.organize_clusters(
                embeddings,
                cluster_labels,
                sentence_list
            )
        perf_tracker.record_stage("3. Clustering", timer.duration)

        # Limit to max_clusters (already sorted by size)
        if len(organized_clusters) > self.max_clusters:
            logger.info(
                "Limiting from %s to %s clusters",
                len(organized_clusters), self.max_clusters
            )
            cluster_ids = list(organized_clusters.keys())[:self.max_clusters]
            organized_clusters = {k: organized_clusters[k] for k in cluster_ids}

        # Step 4: Analyze sentiment for each cluster (parallel)
        logger.info("Step 3/4: Analyzing sentiment (parallel)")
        with PerformanceTimer("4. Sentiment Analysis") as timer:
            cluster_sentiments = asyncio.run(
                self._analyze_sentiments_parallel(organized_clusters)
            )
        perf_tracker.record_stage("4. Sentiment Analysis", timer.duration)

        # Step 5: Generate insights and titles (parallel + combined)
        logger.info("Step 4/4: Generating insights and titles (parallel)")
        with PerformanceTimer("5. Insight Generation") as timer:
            clusters = asyncio.run(self._generate_standalone_insights_parallel(
                organized_clusters,
                cluster_sentiments,
                theme
            ))
        perf_tracker.record_stage("5. Insight Generation", timer.duration)

        logger.info(
            "Standalone analysis complete. Generated %s clusters",
            len(clusters)
        )

        # Log performance summary
        perf_tracker.log_summary()

        return StandaloneAnalysisResponse(clusters=clusters)

    def analyze_comparative(
        self,
        survey_title: str,
        theme: str,
        baseline: List[SentenceInput],
        comparison: List[SentenceInput]
    ) -> ComparativeAnalysisResponse:
        """
        Perform comparative analysis between two datasets.

        Args:
            survey_title: Title of the survey/dataset
            theme: Overall theme of the analysis
            baseline: List of baseline sentence inputs
            comparison: List of comparison sentence inputs

        Returns:
            ComparativeAnalysisResponse with clusters
        """
        logger.info(
            "Starting comparative analysis: survey='%s', "
            "theme='%s', baseline=%s, comparison=%s",
            survey_title, theme, len(baseline), len(comparison)
        )

        # Initialize performance tracker
        perf_tracker = PerformanceTracker("Comparative Analysis")

        # Step 1: Prepare sentence data
        with PerformanceTimer("1. Data Preparation") as timer:
            baseline_data = [
                {'sentence': item.sentence, 'id': item.id}
                for item in baseline
            ]
            comparison_data = [
                {'sentence': item.sentence, 'id': item.id}
                for item in comparison
            ]
        perf_tracker.record_stage("1. Data Preparation", timer.duration)

        # Step 2: Generate embeddings
        logger.info("Step 1/4: Generating embeddings")
        with PerformanceTimer("2. Embedding Generation") as timer:
            baseline_embedding_data = self.embedding_service.generate_embeddings_with_metadata(
                baseline_data
            )
            comparison_embedding_data = self.embedding_service.generate_embeddings_with_metadata(
                comparison_data
            )

            # Convert to arrays
            baseline_embeddings = np.array([
                data['embedding'] for data in baseline_embedding_data.values()
            ])
            comparison_embeddings = np.array([
                data['embedding'] for data in comparison_embedding_data.values()
            ])

            baseline_list = [data for data in baseline_embedding_data.values()]
            comparison_list = [data for data in comparison_embedding_data.values()]
        perf_tracker.record_stage("2. Embedding Generation", timer.duration)

        # Step 3: Perform comparative clustering
        logger.info("Step 2/4: Clustering sentences (comparative)")
        with PerformanceTimer("3. Comparative Clustering") as timer:
            organized_clusters = self.clustering_service.cluster_comparative_data(
                baseline_embeddings,
                comparison_embeddings,
                baseline_list,
                comparison_list
            )
        perf_tracker.record_stage("3. Comparative Clustering", timer.duration)

        # Limit to max_clusters
        if len(organized_clusters) > self.max_clusters:
            logger.info(
                "Limiting from %s to %s clusters",
                len(organized_clusters), self.max_clusters
            )
            cluster_ids = list(organized_clusters.keys())[:self.max_clusters]
            organized_clusters = {k: organized_clusters[k] for k in cluster_ids}

        # Step 4: Analyze sentiment for each cluster (parallel)
        logger.info("Step 3/4: Analyzing sentiment (parallel)")
        with PerformanceTimer("4. Sentiment Analysis") as timer:
            cluster_sentiments = asyncio.run(
                self._analyze_sentiments_parallel_comparative(organized_clusters)
            )
        perf_tracker.record_stage("4. Sentiment Analysis", timer.duration)

        # Step 5: Generate insights and titles (parallel + combined)
        logger.info("Step 4/4: Generating insights and titles (parallel)")
        with PerformanceTimer("5. Insight Generation") as timer:
            clusters = asyncio.run(self._generate_comparative_insights_parallel(
                organized_clusters,
                cluster_sentiments,
                theme
            ))
        perf_tracker.record_stage("5. Insight Generation", timer.duration)

        logger.info(
            "Comparative analysis complete. Generated %s clusters",
            len(clusters)
        )

        # Log performance summary
        perf_tracker.log_summary()

        return ComparativeAnalysisResponse(clusters=clusters)
