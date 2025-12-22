# Thematic Assessment Microservice

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture & Infrastructure](#2-architecture--infrastructure)
3. [Technology Stack & ML Choices](#3-engine--technology-stack--ml-choices)
4. [Core Features & Implementation](#4-core-features--implementation)
5. [Performance Optimization](#5-performance-optimization)
6. [Testing Strategy](#6-testing-strategy)
7. [Deployment & DevOps](#7-deployment--devops)
8. [Cost Analysis](#8-cost-analysis)
9. [Improment in future](#9-improment-in-future)

---

## 1. Executive Summary

### Project Overview

Built a **production-ready serverless microservice** that analyzes customer feedback and groups similar sentences into thematic clusters with sentiment analysis and actionable insights.

### Technical Highlights

- **Serverless Architecture**: AWS Lambda + API Gateway for scalability and cost-efficiency
- **ML Pipeline**: Embeddings → Clustering → Sentiment → Insights
- **Container-Based Deployment**: Lambda Container Images for packaging large ML models (~2GB)
- **Infrastructure as Code**: Complete CloudFormation automation
- **Comprehensive Testing**: 84+ unit tests with >80% coverage

---

## 2. Architecture & Infrastructure

### High-Level Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       │ HTTPS + API Key
       ▼
┌─────────────────────────────────┐
│      API Gateway (REST)         │
│  - /analyze endpoint (POST)     │
│  - API Key authentication       │
│  - CORS enabled                 │
└──────────┬──────────────────────┘
           │
           │ 
           ▼
┌─────────────────────────────────┐
│   Lambda Function (Container)   │
│  - Python 3.12                  │
│  - 3008 MB memory               │
│  - 5 min timeout                │
│  - ML models pre-loaded         │
└──────────┬──────────────────────┘
           │
           ├─────► Secrets Manager (OpenAI API Key)
           │
           ├─────► CloudWatch Logs
           │
           └─────► OpenAI API (GPT-4o)
```

### AWS Services Selection & Rationale

#### **AWS Lambda with Container Images**
**Decision:** Use Lambda Container Images instead of ZIP deployment

**Why:**
- ML models are ~2GB uncompressed, exceeds Lambda ZIP limit (250MB)
- Container images support up to 10GB
- Better dependency management for complex ML libraries
- Pre-download models during Docker build (no runtime downloads)

#### **API Gateway (REST API)**
**Decision:** REST API with API Key authentication

**Why:**
- Simple request/response pattern
- Built-in API key management
- Usage plans for rate limiting (future)
- CORS support for web clients

#### **AWS Secrets Manager**
**Decision:** Store OpenAI API key in Secrets Manager

**Why:**
- No hardcoded secrets in code or environment variables
- Automatic rotation support for secure
- IAM-based access control
- Audit trail via CloudTrail

#### **Amazon ECR**
**Decision:** Private ECR repository for container images

**Why:**
- Seamless integration with Lambda
- Versioned image tags for rollback
- Scan for vulnerabilities
- AWS-managed, no external registries needed

### Infrastructure as Code (IaC)

**Tool:** AWS CloudFormation

**What's Automated:**
- ECR
- Lambda function with container image
- API Gateway REST API + deployment + stage
- IAM roles with least-privilege permissions
- Secrets Manager integration
- CloudWatch log groups with 30-day retention
- API Key generation

### Security Best Practices

**No Hardcoded Secrets**: OpenAI API key in Secrets Manager
**Least Privilege IAM**: Lambda role has minimal permissions (Secrets Manager read + CloudWatch logs)
**API Key Authentication**: All API requests require valid `x-api-key` header
**Input Validation**: Pydantic schema validation on all inputs
**TLS Encryption**: HTTPS only via API Gateway
**No Sensitive Logging**: Customer data not logged to CloudWatch

---

## 3. Engine & Technology Stack & ML Choices

### Core Technology Stack

| Component | Technology | Remark |
|-----------|-----------|---------|
| **Runtime** | Python | 3.12(or 3.13) |
| **Compute** | AWS Lambda ||
| **API** | AWS API Gateway ||
| **IaC** | AWS CloudFormation ||
| **Validation** | Pydantic ||
| **Testing** | pytest + moto ||
| **Logging** | Python logging ||

### Engine/ML/NLP Pipeline

```
Input Sentences
    ↓
[1] Embedding Generation
    ↓ sentence-transformers (all-MiniLM-L6-v2)
    ↓ Output: 384-dimensional vectors
    ↓
[2] Clustering
    ↓ HDBSCAN (density-based)
    ↓ Output: Thematic clusters
    ↓
[3] Sentiment Analysis
    ↓ BERTweet (finiteautomata/bertweet-base-sentiment-analysis)
    ↓ Output: positive/negative/neutral per cluster
    ↓
[4] Insight Generation
    ↓ OpenAI GPT-4o (async API)
    ↓ Output: Title + Key Insights
    ↓
Final Clusters with Insights
```

### ML Model Selection & Rationale

#### **Embeddings: sentence-transformers (all-MiniLM-L6-v2)**

**Why this model:**

| Criteria | all-MiniLM-L6-v2 | Alternatives (e.g., BERT-large) |
|----------|------------------|----------------------------------|
| **Size** | ~80MB | ~400MB+ |
| **Speed** | Fast (critical for <10s target) | Slow |
| **Quality** | Good for general text | Slightly better |
| **Deployment** | Fits in Lambda | Tight fit |

#### **Clustering: HDBSCAN**

**Why HDBSCAN over K-Means:**

| Feature | HDBSCAN | K-Means |
|---------|-----------|---------|
| **Cluster Count** | **Auto-determined** | Manual (must specify K) |
| **Noise Handling** | **Identifies outliers** | Forces all points into clusters |
| **Density Variation** | **Handles varying density** | Assumes uniform density |


**Innovation: Noise Point Recovery**

**Problem:** HDBSCAN marks low-density points as "noise" (label = -1), discarding meaningful sentences.

**Solution:** Soft-assign noise points to nearest cluster if similarity ≥ 0.3

**Algorithm:**
1. HDBSCAN hard clustering → clusters + noise points
2. Calculate cluster centroids (mean embeddings)
3. For each noise point:
   - Compute cosine similarity to all centroids
   - Assign to nearest cluster if similarity ≥ 0.3
   - Otherwise: exclude from results

**Impact:**
- Recovers 20-40% of noise points (typical)
- Maintains cluster coherence (0.3 threshold)
- Increases coverage without sacrificing quality

#### **Sentiment Analysis: BERTweet**

**Why BERTweet:**

| Criteria | BERTweet | Generic BERT |
|----------|------------|--------------|
| **Training Data** | **Twitter (social media)** | Books/Wikipedia |
| **Informal Language** | **Excellent** | Poor |
| **Speed** | Fast (BERT-base size) | Similar |
| **Output** | POS/NEU/NEG | Binary or custom |

**Aggregation Logic:**
- Analyze all sentences in cluster
- Majority vote: dominant sentiment wins
- Tie-breaking: average sentiment scores

#### **Insight Generation: OpenAI GPT-4o**

**Why GPT-4o :**

**Optimization:**
- **One-shot generation**: Title + insights in single API call (not separate calls)
- **Async processing**: Parallel cluster processing with `asyncio.gather()`
- **JSON mode**: Structured output for reliable parsing
- **Sentence limiting**: Max 20 sentences per cluster for cost control

---

## 4. Core Features & Implementation

### Feature 1: Standalone Analysis

### Feature 2: Comparative Analysis

### Feature 3: Robust Input Validation

### Feature 4: API Authentication

### Feature 5: Performance Tracking

**Output:**
```
============================================================
Performance Summary: Standalone Analysis
============================================================
  1. Data Preparation                           0.02s (  0.2%)
  2. Embedding Generation                       2.45s ( 24.5%)
  3. Clustering                                 1.23s ( 12.3%)
  4. Sentiment Analysis                         0.89s (  8.9%)
  5. Insight Generation                         5.41s ( 54.1%)
------------------------------------------------------------
  TOTAL                                        10.00s (100.0%)
============================================================
```
---

## 5. Performance Optimization

### Challenge: <10 Second Response Time Target

**Baseline Performance (before optimization):**
- Embedding: 3.5s
- Clustering: 1.5s
- Sentiment: 2.0s (sequential)
- Insights: 8.0s (sequential)
- **Total: ~15s** (exceeds target)

### Optimization 1: Container-Level Model Caching

### Optimization 2: Pre-Downloaded Models in Docker

### Optimization 3: Parallel Sentiment Analysis

### Optimization 4: Async Insight Generation

### Optimization 5: One-Shot Insight Generation

### Optimization 6: Cold Start Mitigation

**Strategy:** Move expensive operations outside handler

**Lambda Lifecycle:**
```
[Container Init] → Load models (10s, once per container)
    ↓
[Handler Invocation 1] → Process request (10s) ← COLD START
    ↓
[Handler Invocation 2] → Process request (5s)  ← WARM
    ↓
[Handler Invocation 3] → Process request (5s)  ← WARM
```

---

## 6. Testing Strategy

### Test Structure

```
tests/
├── unit/                         # Unit tests (isolated)
│   ├── test_embeddings.py        # 10+ tests
│   ├── test_clustering.py        # 12+ tests
│   ├── test_sentiment.py         # 15+ tests
│   ├── test_insights.py          # 9 tests
│   ├── test_lambda_handler.py    # 13+ tests
│   └── test_schemas.py           # 25+ tests
├── integration/                  # Integration tests (E2E)
│   ├── test_standalone_analysis.py
│   ├── test_comparative_analysis.py
│   └── test_lambda_integration.py
└── conftest.py                   # Shared fixtures
```


---

## 7. Deployment & DevOps

### Deployment Architecture

**Deployment Flow:**
```
Local Development
    ↓
[1] Build Docker Image (5-10 min)
    ├─ Install Python dependencies
    ├─ Pre-download ML models
    └─ Copy application code
    ↓
[2] Push to Amazon ECR
    ├─ Authenticate Docker to ECR
    ├─ Tag image
    └─ Push to repository
    ↓
[3] Deploy CloudFormation Stack
    ├─ Create/update Lambda function
    ├─ Create API Gateway
    ├─ Configure IAM roles
    └─ Set up CloudWatch logging
    ↓
Production Environment
```


## 8. Cost Analysis

### Monthly Cost Estimate (1000 requests/month)

| Service | Usage | Unit Cost | Total |
|---------|-------|-----------|-------|
| **Lambda** | 1000 invocations × 10s × 10240 MB | $0.0000166667/GB-second | $1.50 |
| **API Gateway** | 1000 requests | $0.0000035/request | $0.004 |
| **Secrets Manager** | 1 secret | $0.40/secret/month | $0.40 |
| **ECR** | 5 GB storage | $0.10/GB/month | $0.50 |
| **CloudWatch Logs** | 1 GB ingested | $0.50/GB | $0.50 |
| **Data Transfer** | 10 MB out | $0.09/GB | $0.001 |
| **OpenAI API** | 1000 requests × $0.01/request | GPT-4o pricing | $10.00 |
| **Total (AWS)** | | | **$2.90/month** |
| **Total (with OpenAI)** | | | **$12.90/month** |


## 9. Improment in future

### CI/CD Pipeline (Future Enhancement)

**Proposed Architecture:**

```
GitHub Repository
    ↓
GitHub Actions / AWS CodePipeline
    ↓
[1] Build Stage
    ├─ Checkout code
    ├─ Run pytest (unit + integration)
    ├─ Check coverage (>80%)
    └─ Build Docker image
    ↓
[2] Deploy to Staging
    ├─ Push image to ECR
    ├─ Deploy CloudFormation (staging)
    └─ Run smoke tests
    ↓
[3] Manual Approval (for production)
    ↓
[4] Deploy to Production
    ├─ Deploy CloudFormation (prod)
    ├─ Run smoke tests
    └─ Monitor for errors
```


### ML on Clustering and Sentiment Analysis(Fine-Tuned)
  
### Performance on Lambda

- still cold start, but ping to keep it warm
- Provisioned Concurrency: Keep containers warm (costs more)

### Same sentences re-embedded on every analysis
- Cache embeddings in Redis or DynamoDB

### Batch Processing/Distribution

**Architecture:**
```
API Gateway → Lambda (validate + split)
    ↓
SQS Queue (batches of 100 sentences)
    ↓
Lambda (process batches in parallel)
    ↓
Aggregate results (need more research)
    ↓
Notify client
```


### ECS/AWS SageMaker
Model on S3
Inference Docker Image
Utilize GPU
All warm on-line
Auto Scaling


**Cost Comparison:**

| Service | Configuration | Cost/Month (1000 requests) |
|---------|---------------|---------------------------|
| **Lambda (current)** | 10240 MB, 10s avg | $1.54 |
| **SageMaker** | ml.m5.large (CPU, 24/7) | ~$86 |
| **Lambda(PC)** | ml.m5.large (CPU, 24/7) | ~$86 |

Time for which Provisioned Concurrency is enabled: 720 hours = 2592000 seconds
1 concurrency x 2,592,000 seconds x 10 GB x 0.000005236 USD = 135.72 USD (Provisioned Concurrency charges)
1,000 requests x 12,000 ms x 0.001 ms x 0.0000122173 USD = 1.47 USD (monthly compute charges)
Lambda costs for Provisioned Concurrency (monthly): 137.19 USD


**When to use SageMaker:**
- High request volume
- External Client
- Quick response
