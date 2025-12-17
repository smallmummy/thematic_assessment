# CLAUDE.md - Project Context for AI Assistant

**Version:** 1.0
**Last Updated:** 2025-12-17
**Purpose:** Provide comprehensive context for Claude Code (or other AI assistants) working on this project

---

## Project Overview

This is a **serverless text analysis microservice** that clusters customer feedback sentences into thematic groups with sentiment analysis and actionable insights. Built as a take-home assessment, it represents a real-world production system for analyzing customer feedback at scale.

### Key Facts
- **Time Constraint**: Original task scoped for 4 hours (not meant to be completed to production standards in that time)
- **Primary Focus**: Architecture, infrastructure, testing strategy, and deployment approach (not ML/AI quality)
- **Language**: Python 3.11+
- **Platform**: AWS Lambda + API Gateway (serverless)
- **Main Use Case**: Automated thematic analysis of customer feedback

---

## Core Functionality

### What This Service Does
1. **Accepts** structured JSON input containing sentences with IDs
2. **Clusters** semantically similar sentences using ML embeddings
3. **Analyzes** sentiment for each cluster (positive/negative/neutral)
4. **Generates** actionable insights with markdown formatting
5. **Returns** organized clusters ready for business stakeholders

### Two Analysis Modes

#### 1. Standalone Analysis
- Input: Single dataset of sentences
- Output: Thematic clusters with insights
- Use case: "What are customers saying about our mobile app?"

#### 2. Comparative Analysis (Extra Feature)
- Input: Two datasets (baseline vs comparison)
- Output: Clusters showing similarities and differences
- Use case: "How has feedback changed from Q1 to Q2?"

---

## Technical Architecture

### Technology Stack

**Core Infrastructure:**
- AWS Lambda (compute)
- API Gateway (REST endpoint)
- CloudFormation (Infrastructure as Code)
- Secrets Manager (API key storage)
- CloudWatch (logging/monitoring)

**ML/NLP Pipeline:**
1. **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
2. **Clustering**: HDBSCAN (density-based, auto-determines cluster count)
3. **Sentiment**: transformers library with pre-trained model
4. **Insights**: OpenAI GPT-4o API

**Supporting Libraries:**
- pydantic (validation)
- pytest (testing)
- boto3 (AWS SDK)
- numpy, scikit-learn (data processing)

### Data Flow
```
Client Request
    ↓
API Gateway (authentication)
    ↓
Lambda Handler (validation)
    ↓
[1] Generate embeddings for all sentences
    ↓
[2] Cluster with HDBSCAN
    ↓
[3] Analyze sentiment per cluster
    ↓
[4] Generate insights with GPT-4o
    ↓
Format JSON response
    ↓
Return to client
```

---

## Input/Output Specifications

### Standalone Input
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

**Important Data Constraints:**
- Multiple sentences can share the same `id` (they come from the same comment)
- Each `id` should appear maximum once per cluster in output
- Minimum 1 sentence, recommended max 10,000 per request

### Standalone Output
```json
{
  "clusters": [
    {
      "title": "Specific Sub-theme Title",
      "sentiment": "positive|negative|neutral",
      "sentences": ["id1", "id2"],
      "keyInsights": [
        "Insight with **bold** emphasis",
        "Another **actionable** insight"
      ]
    }
  ]
}
```

**Business Rules:**
- Minimum 3 sentences per cluster (configurable)
- Maximum 20 clusters per response
- Each sentence ID appears in max one cluster
- Insights must be specific, actionable, evidence-based

### Comparative Input/Output
See README.md or PRD.md for full comparative format. Key differences:
- Input has both `baseline` and `comparison` arrays
- Output has `baselineSentences` and `comparisonSentences`
- Output includes `keySimilarities` and `keyDifferences` instead of `keyInsights`

---

## Performance Requirements

### Target Metrics
- **Response Time**: < 10 seconds for 1000 sentences
- **Cold Start**: < 3 seconds for Lambda initialization
- **Availability**: 99.9% uptime
- **Test Coverage**: > 80%

### Scalability Considerations

**Horizontal Scalability** (concurrent requests):
- Lambda auto-scales for concurrent users
- Minimum 10 concurrent executions supported
- Each request processed independently

**Vertical Scalability** (dataset size):
- Small (<100 sentences): <3s, 512MB-1GB memory
- Medium (100-1000): <10s, 2GB memory
- Large (1000-10000): >10s acceptable, 4GB+ memory

---

## Key Files in This Project

### Core Code
- `lambda_handler.py` - Main Lambda entry point
- `map_sentences.py` - Core text analysis logic
- `src/` - Source code modules

### Configuration & Infrastructure
- `cloudformation.yaml` - AWS infrastructure definition
- `requirements.txt` - Python dependencies
- `.env.example` - Environment variables template
- `deployment-config.example.sh` - Deployment configuration
- `Dockerfile` - Container image for Lambda

### Documentation
- `README.md` - Original assessment requirements
- `PRD.md` - Detailed product requirements
- `DEPLOYMENT.md` - Deployment instructions
- `TESTING.md` - Testing strategy and guide
- `EVALUATION.md` - Assessment criteria
- `MAPPING_GUIDE.md` - Sentence mapping logic
- `PERFORMANCE_OPTIMIZATION.md` - Performance tuning
- `PERFORMANCE_TRACKING.md` - Performance monitoring
- `COLD_START_FIX.md` - Cold start optimization

### Testing
- `tests/` - Test directory
- `test_lambda_handler.py` - Lambda handler tests
- `test_standalone.py` - Standalone analysis tests
- `test_comparative.py` - Comparative analysis tests
- `pytest.ini` - Pytest configuration
- `run_tests.sh` - Test execution script

### Sample Data
- `data/input_example.json` - Standalone analysis sample
- `data/input_comparison_example.json` - Comparative analysis sample

### Deployment Scripts
- `deploy.sh` - Deployment automation
- `exec_rebuild_lambda.sh` - Lambda rebuild script

---

## Development Guidelines

### When Making Changes

1. **Read First**: Never modify code without reading the file first
2. **Test Coverage**: Maintain >80% coverage for new code
3. **Avoid Over-Engineering**:
   - Don't add unrequested features
   - Don't refactor surrounding code unnecessarily
   - Keep solutions focused and simple
4. **Security**: Watch for injection vulnerabilities, validate all input
5. **Performance**: Consider Lambda execution time and memory costs

### Testing Approach
- **Unit Tests**: Mock external dependencies (OpenAI, AWS services)
- **Integration Tests**: Use moto for AWS SDK mocking
- **E2E Tests**: Use provided sample data files
- **Performance Tests**: Verify <10s response time target

### Common Pitfalls to Avoid
- Don't forget ID deduplication (same comment can have multiple sentences)
- Don't assume unlimited processing time (Lambda has 15min max)
- Don't ignore cold start optimization (first request after deploy)
- Don't skip input validation (protect against malformed data)
- Don't log sensitive customer data (privacy concern)

---

## Important Design Decisions

### Why HDBSCAN?
- Auto-determines optimal cluster count (vs K-means requiring k)
- Handles varying cluster sizes well
- Identifies noise/outliers naturally
- Better for text data with uneven cluster distributions

### Why sentence-transformers (all-MiniLM-L6-v2)?
- Fast inference (critical for <10s target)
- Runs locally in Lambda (no external API)
- Good quality for general text similarity
- Small model size (fits in Lambda deployment package)

### Why OpenAI GPT-4o for Insights?
- High-quality, context-aware insight generation
- Handles markdown formatting naturally
- Better than template-based approaches for actionable insights
- Trade-off: External API cost vs quality

### Comparative Analysis Architecture Decision
See PRD.md section 11.1: "Comparative Clustering: Cluster together or separately then align?"
- Current approach: Cluster combined dataset, then separate by source
- Alternative: Cluster separately, then align (future consideration)

---

## Known Constraints & Limitations

### Technical Limits
- Lambda timeout: 15 minutes max (AWS limit)
- Lambda memory: 10GB max (AWS limit)
- Lambda package: 250MB unzipped (AWS limit)
- API Gateway payload: 10MB max (AWS limit)
- OpenAI rate limits apply

### Model Limitations
- Embeddings are 384-dimensional (model constraint)
- HDBSCAN performance degrades >50k sentences
- Sentiment model not domain-specific (general pre-trained)
- Insight quality depends on prompt engineering

### Cost Considerations
- Lambda: Memory × execution time
- OpenAI API: Per-token pricing
- API Gateway: Per-request cost

---

## Environment Variables

Required environment variables (see `.env.example`):
- `OPENAI_API_KEY` - Stored in AWS Secrets Manager
- Other configuration as needed

---

## Deployment Process

Quick reference (see `DEPLOYMENT.md` for details):
1. Install dependencies in Lambda-compatible environment
2. Package code + dependencies
3. Upload to S3 or build container image
4. Deploy CloudFormation stack
5. Store API keys in Secrets Manager
6. Test endpoint with sample data
7. Configure monitoring/alarms

---

## When Working on This Project

### Questions to Ask Before Starting
1. "Have I read the relevant source files?"
2. "Do I understand the data constraints (ID deduplication, clustering rules)?"
3. "Am I considering performance implications (Lambda costs, execution time)?"
4. "Have I checked the test files to understand expected behavior?"
5. "Am I maintaining the existing code style and architecture?"

### Where to Find Information
- **Architecture questions**: PRD.md section 5
- **API specs**: PRD.md section 3.4, README.md
- **Testing strategy**: PRD.md section 6, TESTING.md
- **Deployment**: DEPLOYMENT.md
- **Performance targets**: PRD.md section 4.1
- **Sample requests**: `data/` directory

### Common Tasks
- **Add new test**: Check `tests/` directory for patterns
- **Modify clustering**: Look at `map_sentences.py`
- **Update API validation**: Check `lambda_handler.py`
- **Change infrastructure**: Edit `cloudformation.yaml`
- **Optimize performance**: See PERFORMANCE_OPTIMIZATION.md

---

## Success Criteria

### Must Have (Core Requirements)
- ✅ Accepts both standalone and comparative inputs
- ✅ Returns clusters with titles, sentiment, insights
- ✅ Handles ID deduplication correctly
- ✅ Response time <10s for 1000 sentences
- ✅ >80% test coverage
- ✅ Deployed via CloudFormation
- ✅ API authentication working
- ✅ Clear error handling

### Good to Have (Assessment Criteria)
- Well-documented architecture decisions
- Comprehensive testing strategy
- Production-ready error handling
- Monitoring and logging setup
- Clear deployment documentation
- Discussion of trade-offs and alternatives

---

## Assessment Context

This is a **take-home interview task** evaluated on:
1. **Architecture & Infrastructure** - AWS service selection, IaC, security
2. **Testing Strategy** - Coverage, approach, test data management
3. **Deployment & DevOps** - CI/CD thinking, monitoring, documentation
4. **Code Quality** - Readability, error handling, organization
5. **ML Implementation** - Clustering approach, handling edge cases

**NOT primarily evaluated on:**
- Perfect ML model results (quality of clustering/sentiment)
- Complete feature implementation in 4 hours
- Production-scale optimization

**Encouraged to use AI tools** (ChatGPT, Claude, Copilot) but should demonstrate:
- Personal architectural reasoning
- Testing strategy decisions
- Infrastructure choices
- Problem-solving when AI suggestions don't work

---

## Quick Reference

### Key Commands
```bash
# Run tests
./run_tests.sh

# Deploy to AWS
./deploy.sh

# Rebuild Lambda
./exec_rebuild_lambda.sh
```

### Key Endpoints
- **POST /analyze** - Main analysis endpoint
- Authentication: API Key via `x-api-key` header

### Response Codes
- 200: Success
- 400: Invalid input
- 401: Unauthorized
- 413: Payload too large
- 500: Internal error
- 504: Timeout

---

## Notes for AI Assistants

### This Document's Purpose
This file helps you (Claude Code or other AI) understand the project without reading every file. It provides:
- Context on what matters most (architecture > ML quality)
- Common patterns and conventions
- Where to find specific information
- What to watch out for

### When User Asks You To...
- **"Add a feature"**: Check if it aligns with core requirements first
- **"Fix a bug"**: Read relevant source and test files first
- **"Improve performance"**: Consider Lambda costs, not just speed
- **"Add tests"**: Follow existing test patterns in `tests/`
- **"Deploy"**: Follow DEPLOYMENT.md closely
- **"Explain something"**: Reference README.md or PRD.md sections

### AI Tool Usage Documentation
If you're helping with this project, document:
- What parts you generated/suggested
- What architectural decisions were human-made
- Where AI suggestions were modified or rejected
- Any limitations or trade-offs in AI-generated code

---

## Related Documentation

For deeper information, see:
- [README.md](README.md) - Original assessment requirements
- [PRD.md](PRD.md) - Comprehensive product requirements (THIS IS KEY)
- [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment guide
- [TESTING.md](TESTING.md) - Testing strategy
- [EVALUATION.md](EVALUATION.md) - Assessment criteria
- [PERFORMANCE_OPTIMIZATION.md](PERFORMANCE_OPTIMIZATION.md) - Performance tuning

---

**Last Updated:** 2025-12-17
**Maintained By:** Project team
**Feedback:** Update this file as the project evolves