"""
Temporary test script to validate comparative analysis workflow.

Usage:
    1. Create .env file with OPENAI_API_KEY
    2. Activate virtual environment: source venv/bin/activate
    3. Install dependencies: pip install -r requirements.txt
    4. Run: python test_comparative.py
"""

import os
import json
import logging
from dotenv import load_dotenv

from src.models.schemas import SentenceInput, ComparativeAnalysisRequest
from src.services.text_analyzer import TextAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_data(file_path: str) -> dict:
    """Load test data from JSON file."""
    logger.info(f"Loading test data from: {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def main():
    """Run comparative analysis test."""
    # Load environment variables
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')

    if not openai_api_key:
        logger.error("OPENAI_API_KEY not found in environment variables!")
        logger.error("Please create a .env file with your OpenAI API key")
        return

    logger.info("="*80)
    logger.info("Starting Comparative Analysis Test")
    logger.info("="*80)

    # Load test data
    test_data_path = "data/input_comparison_example.json"
    raw_data = load_test_data(test_data_path)

    # For testing, let's use a smaller subset (first 30 sentences each)
    # This will make testing faster and cheaper
    logger.info(f"Total baseline sentences: {len(raw_data['baseline'])}")
    logger.info(f"Total comparison sentences: {len(raw_data['comparison'])}")
    logger.info("Using first 30 sentences from each dataset for testing")

    baseline_subset = raw_data['baseline'][:30]
    comparison_subset = raw_data['comparison'][:30]

    # Parse into Pydantic models
    logger.info("Parsing input data...")
    baseline = [SentenceInput(**item) for item in baseline_subset]
    comparison = [SentenceInput(**item) for item in comparison_subset]

    request = ComparativeAnalysisRequest(
        surveyTitle=raw_data.get('surveyTitle', 'Test Survey'),
        theme=raw_data.get('theme', 'test'),
        baseline=baseline,
        comparison=comparison
    )

    logger.info(f"Survey Title: {request.surveyTitle}")
    logger.info(f"Theme: {request.theme}")
    logger.info(f"Number of baseline sentences: {len(request.baseline)}")
    logger.info(f"Number of comparison sentences: {len(request.comparison)}")

    # Initialize TextAnalyzer
    logger.info("\nInitializing TextAnalyzer...")
    analyzer = TextAnalyzer(
        openai_api_key=openai_api_key,
        min_cluster_size=3,
        max_clusters=10
    )

    # Run analysis
    logger.info("\nRunning comparative analysis...")
    logger.info("This may take a few minutes...")

    try:
        response = analyzer.analyze_comparative(
            survey_title=request.surveyTitle,
            theme=request.theme,
            baseline=request.baseline,
            comparison=request.comparison
        )

        logger.info("\n" + "="*80)
        logger.info("ANALYSIS COMPLETE!")
        logger.info("="*80)

        # Display results
        logger.info(f"\nFound {len(response.clusters)} clusters:\n")

        for i, cluster in enumerate(response.clusters, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Cluster {i}: {cluster.title}")
            logger.info(f"{'='*60}")
            logger.info(f"Sentiment: {cluster.sentiment}")
            logger.info(f"Baseline sentence IDs: {len(cluster.baselineSentences)}")
            logger.info(f"Comparison sentence IDs: {len(cluster.comparisonSentences)}")

            logger.info(f"\nKey Similarities:")
            for j, similarity in enumerate(cluster.keySimilarities, 1):
                logger.info(f"  {j}. {similarity}")

            logger.info(f"\nKey Differences:")
            for j, difference in enumerate(cluster.keyDifferences, 1):
                logger.info(f"  {j}. {difference}")

        # Save results to file
        output_file = "test_output_comparative.json"
        with open(output_file, 'w') as f:
            json.dump(response.model_dump(), f, indent=2)

        logger.info(f"\n{'='*80}")
        logger.info(f"Results saved to: {output_file}")
        logger.info(f"{'='*80}")

    except Exception as e:
        logger.error(f"\nError during analysis: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
