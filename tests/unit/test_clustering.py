"""
Unit tests for ClusteringService.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.services.clustering import ClusteringService


class TestClusteringService:
    """Test ClusteringService class."""

    def test_initialization(self):
        """Test service initialization."""
        service = ClusteringService(
            min_cluster_size=5,
            min_samples=2,
            cluster_selection_epsilon=0.1,
            metric='cosine'
        )

        assert service.min_cluster_size == 5
        assert service.min_samples == 2
        assert service.cluster_selection_epsilon == 0.1
        assert service.metric == 'cosine'

    def test_initialization_defaults(self):
        """Test service initialization with defaults."""
        service = ClusteringService()

        assert service.min_cluster_size == 3
        assert service.min_samples == 1
        assert service.cluster_selection_epsilon == 0.0
        assert service.metric == 'euclidean'

    @patch('src.services.clustering.hdbscan.HDBSCAN')
    def test_cluster_embeddings_success(self, mock_hdbscan):
        """Test successful clustering."""
        # Setup mock
        mock_clusterer = Mock()
        mock_clusterer.fit_predict.return_value = np.array([0, 0, 1, 1, -1])
        mock_hdbscan.return_value = mock_clusterer

        service = ClusteringService()
        embeddings = np.random.rand(5, 10)

        labels = service.cluster_embeddings(embeddings)

        # Verify HDBSCAN was initialized correctly
        mock_hdbscan.assert_called_once_with(
            min_cluster_size=3,
            min_samples=1,
            cluster_selection_epsilon=0.0,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )

        # Verify fit_predict was called
        mock_clusterer.fit_predict.assert_called_once()
        np.testing.assert_array_equal(labels, np.array([0, 0, 1, 1, -1]))

    def test_cluster_embeddings_empty_array(self):
        """Test that empty array raises ValueError."""
        service = ClusteringService()
        empty_embeddings = np.array([])

        with pytest.raises(ValueError) as exc_info:
            service.cluster_embeddings(empty_embeddings)
        assert "Cannot cluster empty embeddings array" in str(exc_info.value)

    @patch('src.services.clustering.hdbscan.HDBSCAN')
    def test_cluster_embeddings_all_noise(self, mock_hdbscan):
        """Test clustering when all points are noise."""
        mock_clusterer = Mock()
        mock_clusterer.fit_predict.return_value = np.array([-1, -1, -1])
        mock_hdbscan.return_value = mock_clusterer

        service = ClusteringService()
        embeddings = np.random.rand(3, 10)

        labels = service.cluster_embeddings(embeddings)

        assert all(label == -1 for label in labels)

    def test_organize_clusters_success(self):
        """Test successful cluster organization."""
        service = ClusteringService()

        embeddings = np.array([
            [1.0, 0.0],
            [1.1, 0.1],
            [0.0, 1.0],
            [0.1, 1.1]
        ])
        cluster_labels = np.array([0, 0, 1, 1])
        sentence_data = [
            {'id': 'id1', 'sentence': 'Text 1', 'embedding': embeddings[0]},
            {'id': 'id2', 'sentence': 'Text 2', 'embedding': embeddings[1]},
            {'id': 'id3', 'sentence': 'Text 3', 'embedding': embeddings[2]},
            {'id': 'id4', 'sentence': 'Text 4', 'embedding': embeddings[3]}
        ]

        result = service.organize_clusters(embeddings, cluster_labels, sentence_data)

        # Should have 2 clusters
        assert len(result) == 2
        assert 0 in result
        assert 1 in result

        # Check cluster 0
        assert result[0]['size'] == 2
        assert set(result[0]['sentence_ids']) == {'id1', 'id2'}
        assert len(result[0]['sentences']) == 2
        assert result[0]['embeddings'].shape == (2, 2)

        # Check cluster 1
        assert result[1]['size'] == 2
        assert set(result[1]['sentence_ids']) == {'id3', 'id4'}

    def test_organize_clusters_with_noise(self):
        """Test cluster organization with noise points."""
        service = ClusteringService()

        embeddings = np.random.rand(5, 10)
        cluster_labels = np.array([0, 0, -1, 1, 1])  # One noise point
        sentence_data = [
            {'id': f'id{i}', 'sentence': f'Text {i}', 'embedding': embeddings[i]}
            for i in range(5)
        ]

        result = service.organize_clusters(embeddings, cluster_labels, sentence_data)

        # Noise point should be excluded
        assert len(result) == 2
        assert all(label != -1 for label in result.keys())

    def test_organize_clusters_deduplication(self):
        """Test ID deduplication within clusters."""
        service = ClusteringService()

        embeddings = np.random.rand(4, 10)
        cluster_labels = np.array([0, 0, 0, 0])
        sentence_data = [
            {'id': 'id1', 'sentence': 'Text 1', 'embedding': embeddings[0]},
            {'id': 'id1', 'sentence': 'Text 1 duplicate', 'embedding': embeddings[1]},
            {'id': 'id2', 'sentence': 'Text 2', 'embedding': embeddings[2]},
            {'id': 'id1', 'sentence': 'Text 1 another', 'embedding': embeddings[3]}
        ]

        result = service.organize_clusters(embeddings, cluster_labels, sentence_data)

        # Should deduplicate IDs within cluster
        assert len(result) == 1
        assert result[0]['size'] == 2  # Only id1 and id2
        assert result[0]['sentence_ids'] == ['id1', 'id2']  # Order preserved
        # All sentences should be kept
        assert len(result[0]['sentences']) == 4

    def test_organize_clusters_sorted_by_size(self):
        """Test that clusters are sorted by size."""
        service = ClusteringService()

        embeddings = np.random.rand(6, 10)
        cluster_labels = np.array([0, 1, 1, 1, 2, 2])
        sentence_data = [
            {'id': f'id{i}', 'sentence': f'Text {i}', 'embedding': embeddings[i]}
            for i in range(6)
        ]

        result = service.organize_clusters(embeddings, cluster_labels, sentence_data)

        # Get cluster IDs in order
        cluster_ids = list(result.keys())

        # Verify sorted by size (descending)
        sizes = [result[cid]['size'] for cid in cluster_ids]
        assert sizes == sorted(sizes, reverse=True)

    @patch('src.services.clustering.hdbscan.HDBSCAN')
    def test_cluster_comparative_data_success(self, mock_hdbscan):
        """Test comparative clustering."""
        # Setup mock
        mock_clusterer = Mock()
        # 6 total sentences: 3 baseline + 3 comparison
        mock_clusterer.fit_predict.return_value = np.array([0, 0, 1, 0, 1, 1])
        mock_hdbscan.return_value = mock_clusterer

        service = ClusteringService()

        baseline_embeddings = np.array([[1.0, 0.0], [1.1, 0.1], [0.0, 1.0]])
        comparison_embeddings = np.array([[1.05, 0.05], [0.05, 1.05], [0.1, 1.1]])

        baseline_data = [
            {'id': 'b1', 'sentence': 'Baseline 1', 'embedding': baseline_embeddings[0]},
            {'id': 'b2', 'sentence': 'Baseline 2', 'embedding': baseline_embeddings[1]},
            {'id': 'b3', 'sentence': 'Baseline 3', 'embedding': baseline_embeddings[2]}
        ]
        comparison_data = [
            {'id': 'c1', 'sentence': 'Comparison 1', 'embedding': comparison_embeddings[0]},
            {'id': 'c2', 'sentence': 'Comparison 2', 'embedding': comparison_embeddings[1]},
            {'id': 'c3', 'sentence': 'Comparison 3', 'embedding': comparison_embeddings[2]}
        ]

        result = service.cluster_comparative_data(
            baseline_embeddings, comparison_embeddings,
            baseline_data, comparison_data
        )

        # Should have 2 clusters
        assert len(result) == 2

        # Check cluster 0: should have b1, b2, c1
        assert 0 in result
        assert set(result[0]['baseline_sentence_ids']) == {'b1', 'b2'}
        assert set(result[0]['comparison_sentence_ids']) == {'c1'}
        assert len(result[0]['baseline_sentences']) == 2
        assert len(result[0]['comparison_sentences']) == 1

        # Check cluster 1: should have b3, c2, c3
        assert 1 in result
        assert set(result[1]['baseline_sentence_ids']) == {'b3'}
        assert set(result[1]['comparison_sentence_ids']) == {'c2', 'c3'}

    @patch('src.services.clustering.hdbscan.HDBSCAN')
    def test_cluster_comparative_data_with_deduplication(self, mock_hdbscan):
        """Test comparative clustering with duplicate IDs."""
        mock_clusterer = Mock()
        mock_clusterer.fit_predict.return_value = np.array([0, 0, 0, 0])
        mock_hdbscan.return_value = mock_clusterer

        service = ClusteringService()

        baseline_embeddings = np.array([[1.0, 0.0], [1.1, 0.1]])
        comparison_embeddings = np.array([[1.05, 0.05], [1.06, 0.06]])

        baseline_data = [
            {'id': 'same_id', 'sentence': 'Baseline 1', 'embedding': baseline_embeddings[0]},
            {'id': 'same_id', 'sentence': 'Baseline 2', 'embedding': baseline_embeddings[1]}
        ]
        comparison_data = [
            {'id': 'same_id', 'sentence': 'Comparison 1', 'embedding': comparison_embeddings[0]},
            {'id': 'other_id', 'sentence': 'Comparison 2', 'embedding': comparison_embeddings[1]}
        ]

        result = service.cluster_comparative_data(
            baseline_embeddings, comparison_embeddings,
            baseline_data, comparison_data
        )

        # Should deduplicate within each source
        assert len(result) == 1
        assert result[0]['baseline_sentence_ids'] == ['same_id']
        assert set(result[0]['comparison_sentence_ids']) == {'same_id', 'other_id'}
        # But keep all sentences
        assert len(result[0]['baseline_sentences']) == 2
        assert len(result[0]['comparison_sentences']) == 2

    @patch('src.services.clustering.hdbscan.HDBSCAN')
    def test_cluster_comparative_data_one_sided_cluster(self, mock_hdbscan):
        """Test comparative clustering when cluster has only baseline or comparison."""
        mock_clusterer = Mock()
        # Cluster 0: only baseline, Cluster 1: only comparison
        mock_clusterer.fit_predict.return_value = np.array([0, 0, 1, 1])
        mock_hdbscan.return_value = mock_clusterer

        service = ClusteringService()

        baseline_embeddings = np.array([[1.0, 0.0], [1.1, 0.1]])
        comparison_embeddings = np.array([[0.0, 1.0], [0.1, 1.1]])

        baseline_data = [
            {'id': 'b1', 'sentence': 'Baseline 1', 'embedding': baseline_embeddings[0]},
            {'id': 'b2', 'sentence': 'Baseline 2', 'embedding': baseline_embeddings[1]}
        ]
        comparison_data = [
            {'id': 'c1', 'sentence': 'Comparison 1', 'embedding': comparison_embeddings[0]},
            {'id': 'c2', 'sentence': 'Comparison 2', 'embedding': comparison_embeddings[1]}
        ]

        result = service.cluster_comparative_data(
            baseline_embeddings, comparison_embeddings,
            baseline_data, comparison_data
        )

        # Cluster 0: only baseline
        assert len(result[0]['baseline_sentence_ids']) == 2
        assert len(result[0]['comparison_sentence_ids']) == 0
        assert len(result[0]['baseline_sentences']) == 2
        assert len(result[0]['comparison_sentences']) == 0

        # Cluster 1: only comparison
        assert len(result[1]['baseline_sentence_ids']) == 0
        assert len(result[1]['comparison_sentence_ids']) == 2
        assert len(result[1]['comparison_sentences']) == 2
        assert len(result[1]['baseline_sentences']) == 0
