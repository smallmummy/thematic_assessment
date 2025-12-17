"""
Sentiment analysis service using transformers library.
"""
import time
import logging
from typing import List, Dict, Literal
import numpy as np
from transformers import pipeline

logger = logging.getLogger(__name__)


class SentimentService:
    """Service for analyzing sentiment of text."""

    def __init__(
        self,
        model_name: str = "finiteautomata/bertweet-base-sentiment-analysis",
        max_text_length: int = 512
    ):
        """
        Initialize the sentiment analysis service.

        Args:
            model_name: Name of the pre-trained sentiment model
            max_text_length: Maximum text length for truncation (default: 512)
        """
        logger.info("Loading sentiment analysis model: %s", model_name)
        self.model_name = model_name
        self.max_text_length = max_text_length
        self.classifier = pipeline(
            "sentiment-analysis",
            model=model_name,
            top_k=None  # Return all labels with scores
        )
        logger.info("Sentiment analysis model loaded successfully")

    def analyze_batch_sentiment(
        self,
        texts: List[str],
        batch_size: int = 8
    ) -> List[Dict[str, float]]:
        """
        Analyze sentiment for a batch of texts.

        Args:
            texts: List of text strings
            batch_size: Batch size for processing

        Returns:
            List of sentiment dictionaries with 'positive', 'negative', 'neutral', and 'label' keys
        """
        if not texts:
            raise ValueError("Cannot analyze sentiment of empty text list")

        logger.info(
            "Analyzing sentiment for %s texts with batch size %s",
            len(texts), batch_size
        )

        try:
            # Truncate texts to model max length
            truncated_texts = [text[:self.max_text_length] for text in texts]

            results = []
            for i in range(0, len(truncated_texts), batch_size):
                batch = truncated_texts[i:i + batch_size]
                batch_results = self.classifier(batch)

                for result in batch_results:
                    # BERTweet model returns labels: 'POS', 'NEU', 'NEG'
                    # e.g [[{'label': 'NEG', 'score': 0.9774207472801208}, {'label': 'NEU', 'score': 0.018556762486696243}, {'label': 'POS', 'score': 0.00402254331856966}]]
                    # Convert to lowercase and map to our format
                    label_mapping = {
                        'pos': 'positive',
                        'neu': 'neutral',
                        'neg': 'negative'
                    }

                    scores = {}
                    for item in result:
                        label = item['label'].lower()
                        mapped_label = label_mapping.get(label, label)
                        scores[mapped_label] = item['score']

                    positive_score = scores.get('positive', 0.0)
                    negative_score = scores.get('negative', 0.0)
                    neutral_score = scores.get('neutral', 0.0)

                    # Determine label based on highest score
                    max_score = max(positive_score, negative_score, neutral_score)
                    if positive_score == max_score:
                        label = 'positive'
                    elif negative_score == max_score:
                        label = 'negative'
                    else:
                        label = 'neutral'

                    results.append({
                        'positive': positive_score,
                        'negative': negative_score,
                        'neutral': neutral_score,
                        'label': label
                    })

            logger.info(
                "Sentiment analysis complete for %s texts",
                len(results)
            )
            return results

        except Exception as e:
            logger.error("Error in batch sentiment analysis: %s", str(e), exc_info=True)
            raise

    def aggregate_cluster_sentiment(
        self,
        sentences: List[str]
    ) -> Literal["positive", "negative", "neutral"]:
        """
        Aggregate sentiment for a cluster of sentences.

        Args:
            sentences: List of sentence texts in the cluster

        Returns:
            Aggregated sentiment label: 'positive', 'negative', or 'neutral'
        """
        start_time = time.time()

        if not sentences:
            raise ValueError("Cannot aggregate sentiment for empty sentence list")

        logger.info(
            "Aggregating sentiment for cluster with %s sentences",
            len(sentences)
        )

        # Analyze all sentences
        sentiments = self.analyze_batch_sentiment(sentences)

        # Calculate average scores for all three categories
        avg_positive = np.mean([s['positive'] for s in sentiments])
        avg_negative = np.mean([s['negative'] for s in sentiments])
        avg_neutral = np.mean([s['neutral'] for s in sentiments])

        # Count label occurrences
        label_counts = {
            'positive': sum(1 for s in sentiments if s['label'] == 'positive'),
            'negative': sum(1 for s in sentiments if s['label'] == 'negative'),
            'neutral': sum(1 for s in sentiments if s['label'] == 'neutral')
        }

        # Determine cluster sentiment based on majority vote with score tiebreaker
        max_count = max(label_counts.values())
        max_labels = [label for label, count in label_counts.items() if count == max_count]

        if len(max_labels) == 1:
            final_label = max_labels[0]
        else:
            # Tie-breaker: use average scores
            # Find the label with highest average score
            avg_scores = {
                'positive': avg_positive,
                'negative': avg_negative,
                'neutral': avg_neutral
            }
            final_label = max(avg_scores, key=avg_scores.get)

        logger.info(
            "Cluster sentiment: %s "
            "(pos=%s, neg=%s, neu=%s, avg_pos=%.3f, avg_neg=%.3f, avg_neu=%.3f)",
            final_label,
            label_counts['positive'], label_counts['negative'],
            label_counts['neutral'], avg_positive, avg_negative, avg_neutral
        )

        total_time = time.time() - start_time
        logger.info(
            "[PERF] Cluster sentiment aggregation for %s sentences took %.2f seconds",
            len(sentences), total_time
        )

        return final_label
