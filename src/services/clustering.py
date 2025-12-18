"""
Clustering service using HDBSCAN for thematic grouping.
"""

import logging
from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np
import hdbscan
from sklearn.metrics.pairwise import cosine_similarity


logger = logging.getLogger(__name__)


class ClusteringService:
    """Service for clustering sentences into thematic groups."""

    def __init__(
        self,
        min_cluster_size: int = 3,
        min_samples: int = 1,
        cluster_selection_epsilon: float = 0.0,
        metric: str = 'euclidean',
        similarity_threshold: float = 0.3
    ):
        """
        Initialize the clustering service.

        Args:
            min_cluster_size: Minimum number of sentences to form a cluster
            min_samples: Minimum samples in a neighborhood for a point to be core
            cluster_selection_epsilon: Distance threshold for cluster selection
            metric: Distance metric to use
            similarity_threshold: Minimum cosine similarity for soft assignment of noise points (default: 0.3)
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.metric = metric
        self.similarity_threshold = similarity_threshold
        logger.info(
            "Initialized ClusteringService with min_cluster_size=%s, min_samples=%s, metric=%s, similarity_threshold=%s",
            min_cluster_size, min_samples, metric, similarity_threshold
        )

    def _soft_assign_noise_points(
        self,
        embeddings: np.ndarray,
        cluster_labels: np.ndarray
    ) -> np.ndarray:
        """
        Soft-assign noise points to nearest cluster using cosine similarity.

        Args:
            embeddings: numpy array of shape (n_sentences, embedding_dim)
            cluster_labels: array of cluster assignments (-1 for noise)

        Returns:
            Updated cluster_labels array with soft-assigned points
        """
        noise_mask = cluster_labels == -1
        n_noise = noise_mask.sum()

        if n_noise == 0:
            logger.info("No noise points to soft-assign")
            return cluster_labels

        # Calculate cluster centroids from hard-assigned clusters
        unique_clusters = set(cluster_labels[~noise_mask])
        if len(unique_clusters) == 0:
            logger.warning("No valid clusters found for soft assignment")
            return cluster_labels

        centroids = {}
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            centroids[cluster_id] = embeddings[cluster_mask].mean(axis=0)

        # Soft-assign each noise point
        n_assigned = 0
        noise_indices = np.where(noise_mask)[0]

        for idx in noise_indices:
            # Calculate cosine similarity to each centroid
            similarities = {}
            for cluster_id, centroid in centroids.items():
                sim = cosine_similarity(
                    embeddings[idx].reshape(1, -1),
                    centroid.reshape(1, -1)
                )[0][0]
                similarities[cluster_id] = sim

            # Find best cluster
            best_cluster = max(similarities, key=similarities.get)
            best_sim = similarities[best_cluster]

            # Assign if above threshold
            if best_sim >= self.similarity_threshold:
                cluster_labels[idx] = best_cluster
                n_assigned += 1

        logger.info(
            "Soft assignment: %d/%d noise points assigned (%.1f%%), %d remain as noise",
            n_assigned, n_noise, (n_assigned / n_noise * 100) if n_noise > 0 else 0,
            n_noise - n_assigned
        )

        return cluster_labels

    def cluster_embeddings(
        self,
        embeddings: np.ndarray,
    ) -> Tuple[np.ndarray, hdbscan.HDBSCAN]:
        """
        Cluster sentence embeddings using HDBSCAN.

        Args:
            embeddings: numpy array of shape (n_sentences, embedding_dim)

        Returns:
            Tuple of (cluster_labels, clusterer_object)
            - cluster_labels: array of cluster assignments (-1 for noise)
            - clusterer_object: fitted HDBSCAN object with probabilities
        """
        if len(embeddings) == 0:
            raise ValueError("Cannot cluster empty embeddings array")

        logger.info("Clustering %s embeddings", len(embeddings))

        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                cluster_selection_epsilon=self.cluster_selection_epsilon,
                metric=self.metric,
                cluster_selection_method='leaf',  # Excess of Mass
                prediction_data=True  # Enable prediction capabilities
            )

            cluster_labels = clusterer.fit_predict(embeddings)

            # Count hard clusters (excluding noise label -1)
            unique_labels = set(cluster_labels)
            n_clusters = len(unique_labels - {-1})
            n_noise_before = list(cluster_labels).count(-1)
            total_points = len(cluster_labels)

            logger.info(
                "Hard clustering complete. Found %s clusters and %s noise points (%.1f%% in clusters)",
                n_clusters, n_noise_before,
                ((total_points - n_noise_before) / total_points * 100) if total_points > 0 else 0
            )

            # Apply soft assignment for noise points
            cluster_labels = self._soft_assign_noise_points(embeddings, cluster_labels)

            # Count final statistics after soft assignment
            n_noise_after = list(cluster_labels).count(-1)
            n_soft_assigned = n_noise_before - n_noise_after

            logger.info(
                "Final clustering: %s hard + %s soft = %s total assigned (%.1f%%), %s noise (%.1f%%)",
                total_points - n_noise_before, n_soft_assigned,
                total_points - n_noise_after,
                ((total_points - n_noise_after) / total_points * 100) if total_points > 0 else 0,
                n_noise_after,
                (n_noise_after / total_points * 100) if total_points > 0 else 0
            )

            return cluster_labels

        except Exception as e:
            logger.error("Error during clustering: %s", str(e), exc_info=True)
            raise

    def organize_clusters(
        self,
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        sentence_data: List[Dict]
    ) -> Dict[int, Dict]:
        """
        Organize clustering results into structured format.

        Args:
            embeddings: numpy array of embeddings
            cluster_labels: array of cluster assignments
            sentence_data: list of dicts with 'id', 'sentence', 'embedding' keys

        Returns:
            Dictionary mapping cluster_id to cluster information:
            {
                cluster_id: {
                    'sentence_ids': [unique_ids],  # Deduplicated IDs
                    'sentences': [sentence_texts],
                    'embeddings': np.array,
                    'size': int
                }
            }
        """
        clusters = defaultdict(lambda: {
            'sentence_ids': [],
            'sentences': [],
            'embeddings': [],
            'original_ids': []  # Track all IDs including duplicates
        })

        for idx, (label, data) in enumerate(zip(cluster_labels, sentence_data)):
            # Skip noise points (label = -1)
            if label == -1:
                continue

            sentence_id = data['id']
            sentence_text = data['sentence']
            embedding = embeddings[idx]

            clusters[label]['original_ids'].append(sentence_id)
            clusters[label]['sentences'].append(sentence_text)
            clusters[label]['embeddings'].append(embedding)

        # Deduplicate sentence IDs per cluster
        # According to spec: "an id could be in multiple resulting clusters,
        # but should be only once per cluster"
        for cluster_id in clusters:
            # Deduplicate while preserving order
            seen = set()
            unique_ids = []
            for sid in clusters[cluster_id]['original_ids']:
                if sid not in seen:
                    seen.add(sid)
                    unique_ids.append(sid)

            clusters[cluster_id]['sentence_ids'] = unique_ids
            clusters[cluster_id]['embeddings'] = np.array(clusters[cluster_id]['embeddings'])
            clusters[cluster_id]['size'] = len(unique_ids)

            # Remove temporary field
            del clusters[cluster_id]['original_ids']

        # Convert to regular dict and sort by cluster size (largest first)
        result = dict(clusters)
        result = dict(sorted(result.items(), key=lambda x: x[1]['size'], reverse=True))

        logger.info(
            "Organized %s clusters. Sizes: %s",
            len(result), [c['size'] for c in result.values()]
        )

        return result

    def cluster_comparative_data(
        self,
        baseline_embeddings: np.ndarray,
        comparison_embeddings: np.ndarray,
        baseline_data: List[Dict],
        comparison_data: List[Dict]
    ) -> Tuple[Dict[int, Dict], Dict[int, Dict]]:
        """
        Cluster baseline and comparison data together, then separate results.

        Args:
            baseline_embeddings: embeddings for baseline sentences
            comparison_embeddings: embeddings for comparison sentences
            baseline_data: metadata for baseline sentences
            comparison_data: metadata for comparison sentences

        Returns:
            Tuple of (baseline_clusters, comparison_clusters)
            Each cluster dict contains both baseline and comparison sentences
        """
        # Combine embeddings
        all_embeddings = np.vstack([baseline_embeddings, comparison_embeddings])

        # Mark which dataset each sentence comes from
        n_baseline = len(baseline_embeddings)
        n_comparison = len(comparison_embeddings)

        combined_data = []
        for i, data in enumerate(baseline_data):
            combined_data.append({**data, 'source': 'baseline', 'original_index': i})
        for i, data in enumerate(comparison_data):
            combined_data.append({**data, 'source': 'comparison', 'original_index': i})

        logger.info(
            "Clustering comparative data: %s baseline + %s comparison sentences",
            n_baseline, n_comparison
        )

        # Cluster all together
        cluster_labels = self.cluster_embeddings(all_embeddings)

        # Organize and separate by source
        clusters = defaultdict(lambda: {
            'baseline_sentence_ids': [],
            'comparison_sentence_ids': [],
            'baseline_sentences': [],
            'comparison_sentences': [],
            'baseline_embeddings': [],
            'comparison_embeddings': [],
            'baseline_original_ids': [],
            'comparison_original_ids': []
        })

        for idx, (label, data) in enumerate(zip(cluster_labels, combined_data)):
            if label == -1:
                continue

            source = data['source']
            sentence_id = data['id']
            sentence_text = data['sentence']
            embedding = all_embeddings[idx]

            if source == 'baseline':
                clusters[label]['baseline_original_ids'].append(sentence_id)
                clusters[label]['baseline_sentences'].append(sentence_text)
                clusters[label]['baseline_embeddings'].append(embedding)
            else:
                clusters[label]['comparison_original_ids'].append(sentence_id)
                clusters[label]['comparison_sentences'].append(sentence_text)
                clusters[label]['comparison_embeddings'].append(embedding)

        # Deduplicate IDs and finalize structure
        for cluster_id in clusters:
            # Deduplicate baseline IDs
            seen_baseline = set()
            unique_baseline_ids = []
            for sid in clusters[cluster_id]['baseline_original_ids']:
                if sid not in seen_baseline:
                    seen_baseline.add(sid)
                    unique_baseline_ids.append(sid)

            # Deduplicate comparison IDs
            seen_comparison = set()
            unique_comparison_ids = []
            for sid in clusters[cluster_id]['comparison_original_ids']:
                if sid not in seen_comparison:
                    seen_comparison.add(sid)
                    unique_comparison_ids.append(sid)

            clusters[cluster_id]['baseline_sentence_ids'] = unique_baseline_ids
            clusters[cluster_id]['comparison_sentence_ids'] = unique_comparison_ids

            clusters[cluster_id]['baseline_embeddings'] = np.array(
                clusters[cluster_id]['baseline_embeddings']
            ) if clusters[cluster_id]['baseline_embeddings'] else np.array([])

            clusters[cluster_id]['comparison_embeddings'] = np.array(
                clusters[cluster_id]['comparison_embeddings']
            ) if clusters[cluster_id]['comparison_embeddings'] else np.array([])

            clusters[cluster_id]['size'] = (
                len(unique_baseline_ids) + len(unique_comparison_ids)
            )

            # Remove temporary fields
            del clusters[cluster_id]['baseline_original_ids']
            del clusters[cluster_id]['comparison_original_ids']

        # Convert to regular dict and sort by size
        result = dict(clusters)
        result = dict(sorted(result.items(), key=lambda x: x[1]['size'], reverse=True))

        logger.info(
            "Organized %s comparative clusters. Sizes: %s",
            len(result), [c['size'] for c in result.values()]
        )

        return result
