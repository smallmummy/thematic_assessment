"""
Temporary test script to validate standalone analysis workflow.

Usage:
    1. Create .env file with OPENAI_API_KEY
    2. Activate virtual environment: source venv/bin/activate
    3. Install dependencies: pip install -r requirements.txt
    4. Run: python test_standalone.py
"""

import os
import json
import logging
from dotenv import load_dotenv

from src.models.schemas import SentenceInput, StandaloneAnalysisRequest
from src.services.text_analyzer import TextAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
MAX_SENTENCES_FOR_TEST = 5000


def load_test_data(file_path: str) -> dict:
    """Load test data from JSON file."""
    logger.info(f"Loading test data from: {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def main():
    """Run standalone analysis test."""
    # Load environment variables
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')

    if not openai_api_key:
        logger.error("OPENAI_API_KEY not found in environment variables!")
        logger.error("Please create a .env file with your OpenAI API key")
        return

    logger.info("="*80)
    logger.info("Starting Standalone Analysis Test")
    logger.info("="*80)

    # Load test data
    test_data_path = "data/input_example.json"
    raw_data = load_test_data(test_data_path)

    # For testing, let's use a smaller subset (first 50 sentences)
    # This will make testing faster and cheaper
    logger.info(f"Total sentences in dataset: {len(raw_data['baseline'])}")
    logger.info("Using first 50 sentences for testing")

    baseline_subset = raw_data['baseline'][:MAX_SENTENCES_FOR_TEST]

    # Parse into Pydantic models
    logger.info("Parsing input data...")
    baseline = [SentenceInput(**item) for item in baseline_subset]

    request = StandaloneAnalysisRequest(
        surveyTitle=raw_data.get('surveyTitle', 'Test Survey'),
        theme=raw_data.get('theme', 'test'),
        baseline=baseline
    )

    logger.info(f"Survey Title: {request.surveyTitle}")
    logger.info(f"Theme: {request.theme}")
    logger.info(f"Number of sentences: {len(request.baseline)}")

    # Initialize TextAnalyzer
    logger.info("\nInitializing TextAnalyzer...")
    analyzer = TextAnalyzer(
        openai_api_key=openai_api_key,
        min_cluster_size=3,
        max_clusters=10
    )

    # Run analysis
    logger.info("\nRunning standalone analysis...")
    logger.info("This may take a few minutes...")

    try:
        response = analyzer.analyze_standalone(
            survey_title=request.surveyTitle,
            theme=request.theme,
            baseline=request.baseline
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
            logger.info(f"Number of unique sentence IDs: {len(cluster.sentences)}")
            logger.info(f"\nKey Insights:")
            for j, insight in enumerate(cluster.keyInsights, 1):
                logger.info(f"  {j}. {insight}")

        # Save results to file
        output_file = "test_output_standalone.json"
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
