# Product Requirements Document (PRD)
# Text Analysis Microservice

**Version:** 1.0
**Last Updated:** 2025-12-15
**Status:** Draft

---

## 1. Executive Summary

### 1.1 Overview
Build a serverless microservice that analyzes customer feedback text data and groups similar sentences into thematic clusters. The service performs intelligent clustering, sentiment analysis, and generates actionable insights from structured text input.

### 1.2 Goals
- Accept structured text input via RESTful API
- Perform intelligent clustering of sentences by theme/topic
- Classify sentiment for each thematic cluster
- Generate actionable insights with key patterns highlighted
- Support both standalone analysis and comparative analysis scenarios
- Deploy as a scalable, cost-effective serverless solution on AWS

### 1.3 Success Metrics
- Response time: < 10 seconds for datasets up to 1000 sentences
- Clustering quality: Semantically coherent clusters with clear themes
- API availability: 99.9% uptime
- Cost efficiency: Optimize AWS Lambda execution time and memory usage

---

## 2. Problem Statement

### 2.1 User Need
Organizations receive large volumes of customer feedback across multiple channels. Manually analyzing and categorizing this feedback is time-consuming and inconsistent. There is a need for an automated system that can:
- Identify common themes and sub-themes in customer feedback
- Understand sentiment patterns for each theme
- Generate actionable insights for stakeholders
- Compare feedback across different time periods or data sources

### 2.2 Current Challenges
- Manual categorization is slow and subjective
- Difficult to identify emerging themes in large datasets
- Comparing feedback across different periods requires significant effort
- No standardized way to extract actionable insights

---

## 3. Functional Requirements

### 3.1 Input Format

#### 3.1.1 Standalone Analysis Input
```json
{
  "surveyTitle": "Robinhood App Store",
  "theme": "account",
  "baseline": [
    {
      "sentence": "Example sentence text",
      "id": "unique-sentence-id"
    }
  ]
}
```

**Fields:**
- `surveyTitle` (string, required): Title of the dataset
- `theme` (string, required): Overall theme that all sentences relate to
- `baseline` (array, required): Array of sentence objects
  - `sentence` (string, required): The text content
  - `id` (string, required): Unique identifier from source comment

**Data Constraints:**
- Multiple sentences can share the same `id` (sentences from same comment)
- Each `id` should appear maximum once per cluster in output
- Minimum 1 sentence, recommended maximum 10,000 sentences per request

#### 3.1.2 Comparative Analysis Input
```json
{
  "surveyTitle": "Robinhood App Store",
  "theme": "account",
  "baseline": [
    {
      "sentence": "Example baseline sentence",
      "id": "unique-sentence-id"
    }
  ],
  "comparison": [
    {
      "sentence": "Example comparison sentence",
      "id": "unique-sentence-id"
    }
  ]
}
```

**Additional Fields:**
- `comparison` (array, required for comparative mode): Array of sentence objects with same structure as baseline

### 3.2 Output Format

#### 3.2.1 Standalone Analysis Output
```json
{
  "clusters": [
    {
      "title": "Specific Sub-theme Title",
      "sentiment": "positive|negative|neutral",
      "sentences": ["unique-sentence-id", "unique-sentence-id-2"],
      "keyInsights": [
        "specific **insight1**",
        "specific insight2 **bolded here**",
        "**specific insight3** here"
      ]
    }
  ]
}
```

**Fields:**
- `clusters` (array): Array of thematic cluster objects
  - `title` (string): Concise, descriptive name for the cluster (max 60 characters)
  - `sentiment` (enum): One of "positive", "negative", or "neutral"
  - `sentences` (array of strings): Unique sentence IDs belonging to this cluster
  - `keyInsights` (array of strings): 2-3 insight sentences in markdown format with **bold** emphasis on key points

**Business Rules:**
- Minimum 3 sentences per cluster (configurable)
- Maximum 20 clusters per response (prioritize largest/most significant)
- Each sentence ID appears in maximum one cluster
- Insights should be specific, actionable, and evidence-based

#### 3.2.2 Comparative Analysis Output
```json
{
  "clusters": [
    {
      "title": "Specific Sub-theme Title",
      "sentiment": "positive|negative|neutral",
      "baselineSentences": ["unique-sentence-id"],
      "comparisonSentences": ["unique-sentence-id-2"],
      "keySimilarities": [
        "specific **insight1**",
        "specific insight2 **bolded here**"
      ],
      "keyDifferences": [
        "specific **insight1**",
        "specific insight2 **bolded here**"
      ]
    }
  ]
}
```

**Additional Fields:**
- `baselineSentences` (array of strings): Sentence IDs from baseline dataset
- `comparisonSentences` (array of strings): Sentence IDs from comparison dataset
- `keySimilarities` (array of strings): 2-3 insights about similarities
- `keyDifferences` (array of strings): 2-3 insights about differences

### 3.3 Core Processing Requirements

#### 3.3.1 Text Clustering
- **Embedding Generation**: Convert sentences to vector embeddings using sentence-transformers (all-MiniLM-L6-v2)
- **Clustering Algorithm**: HDBSCAN for density-based clustering
  - Automatically determine optimal number of clusters
  - Handle varying cluster sizes
  - Identify noise/outlier sentences
  - Minimum cluster size: 3 sentences (configurable)
- **Cluster Naming**: Generate descriptive titles using LLM analysis of sentence content
- **Deduplication**: Ensure each sentence ID appears maximum once per cluster

#### 3.3.2 Sentiment Analysis
- **Method**: Use transformers library with pre-trained sentiment model
- **Aggregation**: Determine cluster sentiment by:
  - Calculating sentiment scores for all sentences in cluster
  - Aggregating to cluster-level sentiment (positive/negative/neutral)
  - Handle mixed sentiment clusters appropriately
- **Output**: Single sentiment label per cluster

#### 3.3.3 Insight Generation
- **Method**: Use OpenAI GPT-4o API
- **Requirements**:
  - Generate 2-3 concise, actionable insights per cluster
  - Use markdown with **bold** emphasis on key points
  - Insights should be specific to the cluster content
  - For comparative analysis: generate both similarities and differences
- **Quality**: Insights should be evidence-based and reference patterns in the data

### 3.4 API Requirements

#### 3.4.1 Endpoint Specification
- **Method**: POST
- **Path**: `/analyze`
- **Content-Type**: `application/json`
- **Authentication**: API Key (via `x-api-key` header)

#### 3.4.2 Response Codes
- `200 OK`: Successful analysis
- `400 Bad Request`: Invalid input format or validation errors
- `401 Unauthorized`: Missing or invalid API key
- `413 Payload Too Large`: Request exceeds size limits
- `500 Internal Server Error`: Processing error
- `504 Gateway Timeout`: Processing exceeded timeout limit

#### 3.4.3 Error Response Format
```json
{
  "error": {
    "code": "INVALID_INPUT",
    "message": "Human-readable error message",
    "details": {
      "field": "baseline",
      "issue": "Field is required"
    }
  }
}
```

### 3.5 Input Validation
- Validate JSON structure matches expected schema
- Verify required fields are present
- Check data types for all fields
- Validate array lengths (min/max constraints)
- Sanitize text input to prevent injection attacks
- Validate sentence IDs are non-empty strings

---

## 4. Non-Functional Requirements

### 4.1 Performance
- **Response Time**: < 10 seconds for 1000 sentences
- **Throughput**: Support concurrent requests (Lambda scaling)
- **Cold Start**: < 3 seconds for Lambda cold start
- **Memory**: Optimize for 2GB-4GB Lambda memory allocation

### 4.2 Scalability

#### 4.2.1 Horizontal Scalability (Concurrent Request Handling)
- **Auto-scaling**: Lambda automatically creates new instances for concurrent requests
- **Concurrent Executions**: Support minimum 10 concurrent requests (configurable up to account limits)
- **Burst Handling**: Handle traffic spikes gracefully with reserved concurrency
- **No Request Blocking**: Multiple users can submit analysis requests simultaneously
- **Independence**: Each request is processed in isolation without interference

#### 4.2.2 Vertical Scalability (Large Dataset Handling)
- **Input Size Range**: Handle datasets from 10 to 10,000 sentences in a single request
- **Memory Configuration**: Configure Lambda memory (2GB-4GB) based on dataset size
- **Batch Processing**: Implement efficient batching for embedding generation
- **Timeout Management**: Set appropriate timeout limits (up to 15 minutes for very large datasets)
- **Algorithm Efficiency**: HDBSCAN and embedding generation optimized for large inputs

#### 4.2.3 Scalability Trade-offs
- **Small datasets (< 100 sentences)**: Fast response (< 3s), low memory (512MB-1GB)
- **Medium datasets (100-1000 sentences)**: Target response time < 10 seconds, medium memory (2GB)
- **Large datasets (1000-10000 sentences)**: May exceed 10s target, higher memory (4GB+), possible timeout extension required

### 4.3 Reliability
- **Availability**: 99.9% uptime SLA
- **Error Handling**: Graceful degradation and informative error messages
- **Retry Logic**: Implement exponential backoff for external API calls
- **Timeout Handling**: Set appropriate timeouts for all operations

### 4.4 Security
- **Authentication**: API Gateway with API Key authentication
- **Authorization**: IAM roles with least privilege principle
- **Secrets Management**: Store API keys in AWS Secrets Manager
- **Data Privacy**: No logging of sensitive user data
- **Encryption**: TLS 1.2+ for data in transit
- **Input Sanitization**: Prevent injection attacks

### 4.5 Monitoring & Logging
- **CloudWatch Logs**: Structured logging for all requests
- **Metrics**: Track latency, error rates, cluster counts
- **Alerting**: Set up alarms for error rate thresholds
- **Tracing**: Implement request ID tracking for debugging

### 4.6 Cost Optimization
- **Lambda Memory**: Right-size memory allocation
- **API Calls**: Minimize external API calls (batch where possible)
- **Embeddings**: Use efficient local model (all-MiniLM-L6-v2)
- **Caching**: Consider caching for repeated requests (future enhancement)

---

## 5. Technical Architecture

### 5.1 Technology Stack

#### 5.1.1 Core Technologies
- **Language**: Python 3.11+
- **Framework**: AWS Lambda (serverless)
- **API Gateway**: AWS API Gateway (REST API)
- **Infrastructure as Code**: AWS CloudFormation

#### 5.1.2 ML/NLP Libraries
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Clustering**: HDBSCAN
- **Sentiment Analysis**: transformers library (pre-trained model)
- **Insights Generation**: OpenAI GPT-4o API

#### 5.1.3 Supporting Libraries
- **Data Processing**: numpy, pandas
- **API Client**: openai Python SDK
- **Validation**: pydantic
- **Testing**: pytest, moto, unittest.mock

### 5.2 AWS Services
- **AWS Lambda**: Compute layer for processing
- **API Gateway**: RESTful API endpoint
- **Secrets Manager**: Store OpenAI API keys
- **CloudWatch**: Logging and monitoring
- **IAM**: Access control and permissions

### 5.3 Component Architecture

```
┌─────────────────┐
│   API Gateway   │
│   (REST API)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Lambda Handler │
│  - Validation   │
│  - Routing      │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│     Text Analysis Engine            │
│  ┌──────────────────────────────┐   │
│  │ 1. Embedding Generation      │   │
│  │    (sentence-transformers)   │   │
│  └──────────────────────────────┘   │
│  ┌──────────────────────────────┐   │
│  │ 2. Clustering (HDBSCAN)      │   │
│  └──────────────────────────────┘   │
│  ┌──────────────────────────────┐   │
│  │ 3. Sentiment Analysis        │   │
│  │    (transformers)            │   │
│  └──────────────────────────────┘   │
│  ┌──────────────────────────────┐   │
│  │ 4. Insight Generation        │   │
│  │    (OpenAI GPT-4o)           │   │
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  JSON Response  │
└─────────────────┘
```

### 5.4 Data Flow

#### 5.4.1 Standalone Analysis Flow
1. Client sends POST request to API Gateway
2. API Gateway validates API key and forwards to Lambda
3. Lambda handler validates input schema
4. Text Analysis Engine:
   a. Generate embeddings for all sentences
   b. Perform HDBSCAN clustering
   c. For each cluster:
      - Analyze sentiment using transformers
      - Generate cluster title
      - Create insights using GPT-4o
      - Collect sentence IDs (deduplicated)
5. Format response and return to client

#### 5.4.2 Comparative Analysis Flow
1. Steps 1-3 same as standalone
2. Text Analysis Engine:
   a. Generate embeddings for baseline and comparison sentences
   b. Perform clustering on combined dataset
   c. For each cluster:
      - Identify which sentences are from baseline vs comparison
      - Analyze sentiment
      - Generate cluster title
      - Create similarity insights using GPT-4o
      - Create difference insights using GPT-4o
      - Collect sentence IDs for both groups
3. Format response and return to client

---

## 6. Testing Strategy

### 6.1 Unit Testing
- **Coverage Target**: > 80% code coverage
- **Framework**: pytest
- **Scope**:
  - Input validation functions
  - Embedding generation
  - Clustering logic
  - Sentiment analysis
  - Deduplication logic
  - Response formatting
- **Mocking**: Use unittest.mock for external dependencies

### 6.2 Integration Testing
- **Framework**: pytest + moto
- **Scope**:
  - Lambda handler end-to-end
  - AWS SDK interactions (Secrets Manager)
  - External API calls (OpenAI)
- **Mock AWS Services**: Use moto for AWS service mocking

### 6.3 End-to-End Testing
- **Environment**: Test AWS account
- **Scope**:
  - Full API Gateway → Lambda → Response flow
  - Authentication and authorization
  - Error handling and edge cases
- **Test Data**: Use provided sample data files

### 6.4 Performance Testing
- **Tools**: Apache Bench or Locust
- **Scenarios**:
  - Single request latency
  - Concurrent request handling
  - Large dataset processing (1000+ sentences)
  - Cold start performance

### 6.5 Test Data
- Use provided sample files:
  - `data/input_example.json` (standalone)
  - `data/input_comparison_example.json` (comparative)
  - `data/input_example_2.json` (if exists)
- Create edge case test data:
  - Minimum sentences (3)
  - Maximum sentences (1000+)
  - Single cluster scenario
  - High noise scenario

---

## 7. Deployment Strategy

### 7.1 Infrastructure as Code
- **Tool**: AWS CloudFormation
- **Resources to Define**:
  - Lambda function (with layers for dependencies)
  - API Gateway REST API
  - IAM roles and policies
  - Secrets Manager secrets
  - CloudWatch log groups and alarms

### 7.2 Deployment Environments
- **Development**: Local testing with SAM CLI or LocalStack
- **Staging**: Test AWS account for integration testing
- **Production**: Production AWS account with full monitoring

### 7.3 CI/CD Pipeline (Future Enhancement)
- **Build**: Run tests, lint code
- **Package**: Create Lambda deployment package
- **Deploy**: Update CloudFormation stack
- **Verify**: Run smoke tests

### 7.4 Deployment Checklist
1. Package Python dependencies (including ML models)
2. Create Lambda deployment package (ZIP or container image)
3. Upload to S3 or ECR
4. Deploy CloudFormation stack
5. Store OpenAI API key in Secrets Manager
6. Create API Gateway API key
7. Test endpoint with sample data
8. Configure CloudWatch alarms
9. Document API endpoint and authentication

---

## 8. Dependencies

### 8.1 Python Libraries
```
sentence-transformers>=2.2.0
hdbscan>=0.8.33
transformers>=4.30.0
torch>=2.0.0
openai>=1.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
pydantic>=2.0.0
boto3>=1.28.0
```

### 8.2 External Services
- **OpenAI API**: GPT-4o for insight generation
  - Requires API key
  - Rate limits apply
  - Cost per request

### 8.3 AWS Services
- AWS Lambda
- API Gateway
- Secrets Manager
- CloudWatch
- IAM

---

## 9. Constraints and Limitations

### 9.1 Technical Constraints
- **Lambda Timeout**: Maximum 15 minutes (AWS limit)
- **Lambda Memory**: Maximum 10GB (AWS limit)
- **Lambda Package Size**: 250MB unzipped (AWS limit)
- **API Gateway Payload**: 10MB maximum (AWS limit)
- **OpenAI Rate Limits**: Subject to OpenAI tier limits

### 9.2 Model Limitations
- **Embedding Model**: all-MiniLM-L6-v2 limited to 384-dimensional vectors
- **Clustering**: HDBSCAN performance degrades with very large datasets (>50k sentences)
- **Sentiment Model**: Pre-trained model may not capture domain-specific sentiment
- **Insight Generation**: Quality depends on GPT-4o prompt engineering

### 9.3 Cost Considerations
- Lambda execution cost scales with memory and time
- OpenAI API costs per token (input + output)
- API Gateway requests cost

---

## 10. Future Enhancements

### 10.1 Phase 2 Features
- Caching layer for repeated requests
- Support for additional languages
- Custom sentiment model fine-tuning
- Batch processing API for large datasets
- WebSocket support for real-time progress updates

### 10.2 Optimization Opportunities
- Model quantization for faster inference
- Embedding caching for repeated sentences
- Async processing with SQS for large datasets
- Multi-model comparison for insight quality

### 10.3 Monitoring Enhancements
- Dashboard for cluster quality metrics
- A/B testing framework for algorithm improvements
- User feedback collection for insight quality

---

## 11. Open Questions

### 11.1 To Be Resolved
1. **Cluster Title Generation**: Use LLM or extractive summarization?
2. **Minimum Cluster Size**: Should this be configurable via API?
3. **Maximum Clusters**: Hard limit of 20 or configurable?
4. **Sentiment Threshold**: What confidence threshold for neutral vs positive/negative?
5. **Comparative Clustering**: Cluster together or separately then align?

### 11.2 Assumptions
1. Input text is in English
2. Sentences are already properly segmented
3. IDs are unique across the entire dataset
4. Client can handle response times up to 10 seconds

---

## 12. Success Criteria

### 12.1 Functional Success
- ✅ Accepts both standalone and comparative analysis inputs
- ✅ Returns clusters with titles, sentiment, and insights
- ✅ Handles ID deduplication correctly
- ✅ Generates actionable, high-quality insights

### 12.2 Technical Success
- ✅ Response time < 10 seconds for 1000 sentences
- ✅ Unit test coverage > 80%
- ✅ Successfully deployed via CloudFormation
- ✅ API authentication working correctly
- ✅ Error handling provides clear feedback

### 12.3 Documentation Success
- ✅ API documentation with examples
- ✅ Architecture diagram
- ✅ Deployment guide
- ✅ Testing guide
- ✅ Trade-offs and design decisions documented

---

## 13. Appendix

### 13.1 References
- Original assessment: `README.md`
- Sample data: `data/input_example.json`, `data/input_comparison_example.json`
- HDBSCAN documentation: https://hdbscan.readthedocs.io/
- sentence-transformers: https://www.sbert.net/

### 13.2 Glossary
- **Cluster**: A group of semantically similar sentences
- **Theme**: The overall topic area (e.g., "account", "food")
- **Sub-theme**: Specific topic within a theme (what clusters represent)
- **Embedding**: Vector representation of sentence semantics
- **Sentiment**: Emotional tone (positive/negative/neutral)
- **Insight**: Actionable observation about a cluster

### 13.3 Document History
| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-15 | Initial | Created PRD based on requirements |
