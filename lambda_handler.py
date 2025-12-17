"""
AWS Lambda handler for text analysis microservice.
"""

import json
import os
import logging
from typing import Dict, Any, Optional, TYPE_CHECKING
import boto3
from botocore.exceptions import ClientError

from pydantic import ValidationError

# Configure cache directories for Lambda environment
# Strategy: Pre-downloaded models are in /var/task/.cache/ (read-only, baked in image)
#           Runtime writes go to /tmp/ (writable)
#
# HF_HOME controls where Hugging Face looks for models AND where it writes cache metadata
# We use /tmp for HF_HOME so runtime writes work, but models are still found via
# symlinks or by setting HF_HUB_CACHE to the read-only location
os.environ.setdefault('HF_HOME', '/tmp/huggingface')
os.environ.setdefault('TRANSFORMERS_CACHE', '/tmp/transformers_cache')
os.environ.setdefault('SENTENCE_TRANSFORMERS_HOME', '/tmp/sentence_transformers')
os.environ.setdefault('TORCH_HOME', '/tmp/torch')

# Also set HF_HUB_CACHE to point to pre-downloaded models in the image
# This tells Hugging Face where to FIND models (read-only is OK)
os.environ.setdefault('HF_HUB_CACHE', '/var/task/.cache/huggingface/hub')

from src.models.schemas import (
    StandaloneAnalysisRequest,
    ComparativeAnalysisRequest,
    ErrorResponse
)

# Delay TextAnalyzer import to avoid loading ML models during init phase
if TYPE_CHECKING:
    from src.services.text_analyzer import TextAnalyzer


# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Cache for OpenAI API key (retrieved once per container lifetime)
_openai_api_key_cache: Optional[str] = None

# Cache for TextAnalyzer instance (initialized once per container lifetime)
_text_analyzer_cache: Optional[Any] = None


def get_text_analyzer():
    """
    Get or initialize the TextAnalyzer instance.

    The analyzer is cached at the container level to avoid reloading
    ML models on every invocation. Models are loaded once when the
    container initializes.

    Returns:
        TextAnalyzer instance
    """
    global _text_analyzer_cache

    if _text_analyzer_cache is not None:
        logger.info("Using cached TextAnalyzer instance")
        return _text_analyzer_cache

    # Lazy import - only import TextAnalyzer when actually needed (not during init)
    logger.info("Importing TextAnalyzer module (lazy load)")
    from src.services.text_analyzer import TextAnalyzer

    # Get configuration from environment
    openai_api_key = get_openai_api_key()
    embedding_model = os.environ.get('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    sentiment_model = os.environ.get('SENTIMENT_MODEL', 'finiteautomata/bertweet-base-sentiment-analysis')
    min_cluster_size = int(os.environ.get('MIN_CLUSTER_SIZE', '3'))
    max_clusters = int(os.environ.get('MAX_CLUSTERS', '20'))

    logger.info(
        "Initializing TextAnalyzer: embedding=%s, sentiment=%s, min_cluster=%s, max_clusters=%s",
        embedding_model, sentiment_model, min_cluster_size, max_clusters
    )

    # Initialize analyzer (this loads ML models)
    _text_analyzer_cache = TextAnalyzer(
        openai_api_key=openai_api_key,
        embedding_model=embedding_model,
        sentiment_model=sentiment_model,
        min_cluster_size=min_cluster_size,
        max_clusters=max_clusters
    )

    logger.info("TextAnalyzer initialized and cached")
    return _text_analyzer_cache


def get_openai_api_key() -> str:
    """
    Retrieve OpenAI API key from Secrets Manager or environment variable.

    Supports two modes:
    1. Production: Read from Secrets Manager using OPENAI_API_KEY_SECRET_ARN
    2. Development/Testing: Read directly from OPENAI_API_KEY environment variable

    The key is cached after first retrieval to minimize API calls.

    Returns:
        OpenAI API key string

    Raises:
        ValueError: If API key cannot be retrieved
    """
    global _openai_api_key_cache

    # Return cached value if available
    if _openai_api_key_cache:
        return _openai_api_key_cache

    # Check for direct environment variable (development/testing)
    api_key = os.environ.get('OPENAI_API_KEY')
    if api_key:
        logger.info("Using OpenAI API key from OPENAI_API_KEY environment variable")
        _openai_api_key_cache = api_key
        return api_key

    # Check for Secrets Manager ARN (production)
    secret_arn = os.environ.get('OPENAI_API_KEY_SECRET_ARN')
    if not secret_arn:
        raise ValueError(
            "Neither OPENAI_API_KEY nor OPENAI_API_KEY_SECRET_ARN environment variable is set"
        )

    # Retrieve from Secrets Manager
    try:
        logger.info(f"Retrieving OpenAI API key from Secrets Manager: {secret_arn}")
        session = boto3.session.Session()
        client = session.client('secretsmanager')

        response = client.get_secret_value(SecretId=secret_arn)

        # Parse secret value (could be plain string or JSON)
        if 'SecretString' in response:
            secret = response['SecretString']
            try:
                # Try parsing as JSON first
                secret_dict = json.loads(secret)
                # Look for common key names
                api_key = secret_dict.get('OPENAI_API_KEY') or secret_dict.get('api_key')
                if not api_key:
                    raise ValueError(f"Secret JSON does not contain 'OPENAI_API_KEY' or 'api_key' field")
            except json.JSONDecodeError:
                # Not JSON, treat as plain string
                api_key = secret
        else:
            raise ValueError("Secret does not contain SecretString")

        logger.info("Successfully retrieved OpenAI API key from Secrets Manager")
        _openai_api_key_cache = api_key
        return api_key

    except ClientError as e:
        error_code = e.response['Error']['Code']
        logger.error(f"Failed to retrieve secret from Secrets Manager: {error_code}")
        raise ValueError(f"Cannot retrieve OpenAI API key from Secrets Manager: {error_code}")
    except Exception as e:
        logger.error(f"Unexpected error retrieving secret: {str(e)}")
        raise ValueError(f"Cannot retrieve OpenAI API key: {str(e)}")


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler function for text analysis.

    Args:
        event: API Gateway event containing the request
        context: Lambda context object

    Returns:
        API Gateway response with status code and body
    """
    logger.info("Lambda handler invoked")

    try:
        # Parse request body
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event.get('body', {})

        logger.info("Request body parsed successfully")

        # Get or initialize TextAnalyzer (loads models on first call only)
        try:
            analyzer = get_text_analyzer()
        except ValueError as e:
            logger.error(f"Failed to initialize TextAnalyzer: {str(e)}")
            return create_error_response(
                500,
                "CONFIGURATION_ERROR",
                "Service initialization failed"
            )

        # Determine analysis mode based on presence of 'comparison' field
        is_comparative = 'comparison' in body

        if is_comparative:
            logger.info("Running comparative analysis")
            return handle_comparative_analysis(body, analyzer)
        else:
            logger.info("Running standalone analysis")
            return handle_standalone_analysis(body, analyzer)

    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in request body: %s", str(e))
        return create_error_response(
            400,
            "INVALID_JSON",
            f"Invalid JSON format: {str(e)}"
        )

    except Exception as e:
        logger.error("Unexpected error in lambda handler: %s", str(e), exc_info=True)
        return create_error_response(
            500,
            "INTERNAL_ERROR",
            f"An unexpected error occurred: {str(e)}"
        )


def handle_standalone_analysis(
    body: Dict[str, Any],
    analyzer: Any
) -> Dict[str, Any]:
    """
    Handle standalone analysis request.

    Args:
        body: Request body
        analyzer: Pre-initialized TextAnalyzer instance

    Returns:
        API Gateway response
    """
    try:
        # Validate request
        request = StandaloneAnalysisRequest(**body)
        logger.info(
            "Validated standalone request: survey='%s', theme='%s', sentences=%s",
            request.surveyTitle, request.theme, len(request.baseline)
        )

        # Perform analysis (using cached analyzer)
        result = analyzer.analyze_standalone(
            survey_title=request.surveyTitle,
            theme=request.theme,
            baseline=request.baseline
        )

        logger.info("Standalone analysis completed successfully with %s clusters", len(result.clusters))

        # Return success response
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(result.model_dump())
        }

    except ValidationError as e:
        logger.error("Validation error in standalone analysis: %s", str(e))
        return create_error_response(
            400,
            "VALIDATION_ERROR",
            "Invalid request format",
            {"errors": e.errors()}
        )

    except ValueError as e:
        logger.error("Value error in standalone analysis: %s", str(e))
        return create_error_response(
            400,
            "INVALID_INPUT",
            str(e)
        )

    except Exception as e:
        logger.error("Error in standalone analysis: %s", str(e), exc_info=True)
        return create_error_response(
            500,
            "ANALYSIS_ERROR",
            f"Error during analysis: {str(e)}"
        )


def handle_comparative_analysis(
    body: Dict[str, Any],
    analyzer: Any
) -> Dict[str, Any]:
    """
    Handle comparative analysis request.

    Args:
        body: Request body
        analyzer: Pre-initialized TextAnalyzer instance

    Returns:
        API Gateway response
    """
    try:
        # Validate request
        request = ComparativeAnalysisRequest(**body)
        logger.info(
            "Validated comparative request: survey='%s', theme='%s', baseline=%s, comparison=%s",
            request.surveyTitle, request.theme, len(request.baseline), len(request.comparison)
        )

        # Perform analysis (using cached analyzer)
        result = analyzer.analyze_comparative(
            survey_title=request.surveyTitle,
            theme=request.theme,
            baseline=request.baseline,
            comparison=request.comparison
        )

        logger.info("Comparative analysis completed successfully with %s clusters", len(result.clusters))

        # Return success response
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(result.model_dump())
        }

    except ValidationError as e:
        logger.error("Validation error in comparative analysis: %s", str(e))
        return create_error_response(
            400,
            "VALIDATION_ERROR",
            "Invalid request format",
            {"errors": e.errors()}
        )

    except ValueError as e:
        logger.error("Value error in comparative analysis: %s", str(e))
        return create_error_response(
            400,
            "INVALID_INPUT",
            str(e)
        )

    except Exception as e:
        logger.error("Error in comparative analysis: %s", str(e), exc_info=True)
        return create_error_response(
            500,
            "ANALYSIS_ERROR",
            f"Error during analysis: {str(e)}"
        )


def create_error_response(
    status_code: int,
    error_code: str,
    message: str,
    details: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Create a standardized error response.

    Args:
        status_code: HTTP status code
        error_code: Application error code
        message: Error message
        details: Optional error details

    Returns:
        API Gateway error response
    """
    error_response = ErrorResponse.create(
        code=error_code,
        message=message,
        details=details
    )

    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps(error_response.model_dump())
    }