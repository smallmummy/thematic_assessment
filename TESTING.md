# Testing Summary

## Overview

Comprehensive unit test suite for the thematic assessment microservice, providing high code coverage and validation of all core functionality.

## Test Files Created

### 1. [tests/unit/test_schemas.py](tests/unit/test_schemas.py)
**Coverage**: Pydantic models and validation

**Test Classes**:
- `TestSentenceInput`: Validates sentence input model
  - Valid inputs with whitespace handling
  - Empty sentence/ID validation
  - Whitespace-only inputs

- `TestStandaloneAnalysisRequest`: Validates standalone request model
  - Valid requests
  - Empty baseline validation
  - Maximum sentence limits (10,000)

- `TestComparativeAnalysisRequest`: Validates comparative request model
  - Valid requests with baseline and comparison
  - Empty dataset validation

- `TestStandaloneCluster`: Validates standalone cluster output
  - Valid cluster structure
  - Invalid sentiment handling
  - Insights count validation (2-3 required)

- `TestComparativeCluster`: Validates comparative cluster output
  - Valid cluster with similarities/differences
  - Empty sentence arrays
  - Validation counts for similarities/differences

- `TestAnalysisResponses`: Validates response wrappers
- `TestErrorResponse`: Validates error response formatting

**Total Tests**: 25+ test cases

---

### 2. [tests/unit/test_embeddings.py](tests/unit/test_embeddings.py)
**Coverage**: Embedding generation service

**Test Class**: `TestEmbeddingService`
- Model initialization and configuration
- Successful embedding generation
- Empty list error handling
- Error propagation from model
- Metadata preservation with embeddings
- Custom batch size handling
- Normalization flag verification

**Total Tests**: 10+ test cases

---

### 3. [tests/unit/test_sentiment.py](tests/unit/test_sentiment.py)
**Coverage**: Sentiment analysis service with BERTweet model

**Test Class**: `TestSentimentService`
- Initialization with custom parameters
- Default parameters (BERTweet model, 512 max length)
- Batch sentiment analysis success
- Label mapping (POS/NEU/NEG → positive/neutral/negative)
- All three sentiment scores returned
- Empty list validation
- Text truncation to max_text_length
- Batch processing logic
- Cluster sentiment aggregation (majority vote)
- Tie-breaking with average scores
- Neutral sentiment handling
- Label case handling (lowercase)

**Total Tests**: 15+ test cases

---

### 4. [tests/unit/test_clustering.py](tests/unit/test_clustering.py)
**Coverage**: HDBSCAN clustering service

**Test Class**: `TestClusteringService`
- Initialization with custom/default parameters
- HDBSCAN configuration validation
- Successful clustering with noise detection
- Empty array validation
- All-noise scenario handling
- Cluster organization and structure
- Noise point exclusion
- ID deduplication within clusters
- Sentence preservation (all sentences kept)
- Size-based sorting (largest first)
- Comparative clustering (baseline + comparison)
- Source separation (baseline vs comparison)
- One-sided clusters (only baseline or comparison)

**Total Tests**: 12+ test cases

---

### 5. [tests/unit/test_insights.py](tests/unit/test_insights.py)
**Coverage**: GPT-4o insight generation service (async API)

**Test Class**: `TestInsightService`
- Initialization with AsyncOpenAI client
- Default parameter handling
- Standalone content generation (async, one-shot title + insights)
- Sentence limiting for insights (max_sentences_for_insights)
- No limiting when under max (avoid bias)
- Title truncation (max_title_length)
- Insights count validation (2-3, with fallback)
- Error fallback with default content
- Comparative content generation (async, one-shot title + similarities + differences)
- Sentence limiting for comparison (max_sentences_for_comparison)
- Similarities and differences validation

**Total Tests**: 9 test cases

**Note**: All insight generation uses async methods with JSON mode for efficient one-shot generation of titles and insights together.

---

### 6. [tests/unit/test_lambda_handler.py](tests/unit/test_lambda_handler.py)
**Coverage**: AWS Lambda handler

**Test Classes**:
- `TestLambdaHandler`: Main handler function
  - Standalone analysis success
  - Comparative analysis success
  - Missing API key error (500)
  - Invalid JSON error (400)
  - Validation error (400)
  - Custom configuration from environment variables
  - Analysis error handling (500)

- `TestHandleStandaloneAnalysis`: Standalone handler
  - Success case
  - ValueError handling

- `TestHandleComparativeAnalysis`: Comparative handler
  - Success case

- `TestCreateErrorResponse`: Error response utility
  - Basic error response
  - Error with details
  - CORS headers verification

**Total Tests**: 13+ test cases

---

## Supporting Files

### [pytest.ini](pytest.ini)
Pytest configuration with:
- Test discovery patterns
- Coverage reporting (terminal + HTML)
- Branch coverage enabled
- Async test support
- Test markers (unit, integration, slow, requires_api)
- Warning filters

---

## Test Coverage Summary

| Module | Test File | Tests | Coverage Areas |
|--------|-----------|-------|----------------|
| Schemas | test_schemas.py | 25+ | Input validation, output validation, error handling |
| Embeddings | test_embeddings.py | 10+ | Model mocking, metadata, deduplication |
| Sentiment | test_sentiment.py | 15+ | BERTweet handling, aggregation, tie-breaking |
| Clustering | test_clustering.py | 12+ | HDBSCAN, organization, comparative logic |
| Insights | test_insights.py | 9 | Async generation, limiting, validation |
| Lambda | test_lambda_handler.py | 13+ | API Gateway, routing, error responses |
| **Total** | **6 files** | **84+** | **All core functionality** |

---

## Running the Tests

### Quick Run
```bash
./run_tests.sh
```

### Manual Run
```bash
# All tests with coverage
pytest

# Specific module
pytest tests/unit/test_schemas.py -v

# With coverage report
pytest --cov=src --cov=lambda_handler --cov-report=html
open htmlcov/index.html
```

---

## Continuous Integration

For CI/CD pipelines, use:
```bash
pytest --cov=src --cov=lambda_handler \
       --cov-report=xml \
       --cov-report=term \
       --junitxml=test-results.xml
```

This generates:
- `coverage.xml`: Coverage report for CI tools
- `test-results.xml`: JUnit format for test dashboards
- Terminal output for quick feedback