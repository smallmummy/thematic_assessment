# Deployment Guide

This guide covers deploying the Thematic Assessment Microservice to AWS using Lambda Container Images and CloudFormation.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Architecture Overview](#architecture-overview)
3. [Pre-Deployment Setup](#pre-deployment-setup)
4. [Deployment Steps](#deployment-steps)
5. [Testing the Deployment](#testing-the-deployment)
6. [Monitoring and Troubleshooting](#monitoring-and-troubleshooting)
7. [Updating the Service](#updating-the-service)
8. [Cleanup](#cleanup)

## Prerequisites

### Required Tools

- **AWS CLI**: Version 2.x or higher
  ```bash
  aws --version
  ```

- **Docker**: Version 20.x or higher
  ```bash
  docker --version
  ```

- **AWS Account**: With appropriate permissions for:
  - Lambda
  - API Gateway
  - ECR (Elastic Container Registry)
  - CloudFormation
  - IAM
  - Secrets Manager
  - CloudWatch Logs

### Required Permissions

Your AWS IAM user/role needs the following permissions:
- `cloudformation:*`
- `lambda:*`
- `apigateway:*`
- `ecr:*`
- `iam:CreateRole`, `iam:AttachRolePolicy`, `iam:PassRole`
- `secretsmanager:GetSecretValue`
- `logs:*`

## Architecture Overview

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
           │ AWS_PROXY Integration
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

### Components

1. **Lambda Container Image**: Contains application code + ML models
2. **API Gateway**: REST API with API Key authentication
3. **Secrets Manager**: Securely stores OpenAI API key
4. **CloudWatch**: Logging and monitoring
5. **ECR**: Container image repository

## Pre-Deployment Setup

### 1. Configure API Gateway CloudWatch Logging (One-Time Setup)

**IMPORTANT**: This is a **one-time setup per AWS account** required for API Gateway logging.

If you skip this step, CloudFormation deployment will succeed, but API Gateway logging will be disabled. To enable logging later, complete this setup and uncomment lines 231-232 in `cloudformation.yaml`.

Run these commands once in your AWS account:

```bash
# 1. Create IAM role for API Gateway CloudWatch logging
aws iam create-role \
  --role-name APIGatewayCloudWatchLogsRole \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {
        "Service": "apigateway.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }]
  }'

# 2. Attach CloudWatch Logs policy
aws iam attach-role-policy \
  --role-name APIGatewayCloudWatchLogsRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonAPIGatewayPushToCloudWatchLogs

# 3. Get role ARN
ROLE_ARN=$(aws iam get-role --role-name APIGatewayCloudWatchLogsRole --query 'Role.Arn' --output text)

# 4. Configure API Gateway account (replace us-east-1 with your region)
aws apigateway update-account \
  --region us-east-1 \
  --patch-operations op=replace,path=/cloudwatchRoleArn,value=${ROLE_ARN}

# 5. Verify configuration
aws apigateway get-account --region us-east-1 --query cloudwatchRoleArn --output text
```

**Note**: If the role already exists (e.g., from previous projects), skip step 1 and start from step 3.

After this setup, you can enable API Gateway logging by uncommenting these lines in `cloudformation.yaml`:
```yaml
# LoggingLevel: INFO          # Line 231
# DataTraceEnabled: true      # Line 232
```

### 2. Create OpenAI API Key Secret

Store your OpenAI API key in AWS Secrets Manager:

```bash
# Replace with your actual OpenAI API key
export OPENAI_API_KEY="sk-your-openai-api-key-here"

# Create secret
aws secretsmanager create-secret \
  --name thematic-assessment-openai-key \
  --description "OpenAI API key for thematic assessment service" \
  --secret-string "$OPENAI_API_KEY" \
  --region us-east-1
```

**Important**: Save the secret ARN from the output. You'll need it for deployment.

Example ARN format:
```
arn:aws:secretsmanager:us-east-1:123456789012:secret:thematic-assessment-openai-key-AbCdEf
```

### 3. Configure Deployment Settings

Copy the example configuration and modify with your values:

```bash
cp deployment-config.example.sh deployment-config.sh
```

Edit `deployment-config.sh`:

```bash
#!/bin/bash
export AWS_REGION="us-east-1"
export AWS_ACCOUNT_ID=""  # Leave empty to auto-detect
export PROJECT_NAME="thematic-assessment"
export STACK_NAME="thematic-assessment-stack"

# IMPORTANT: Set this to your Secrets Manager secret ARN
export OPENAI_API_KEY_SECRET_ARN="arn:aws:secretsmanager:us-east-1:123456789012:secret:thematic-assessment-openai-key-XXXXXX"

# Lambda Configuration (defaults are recommended)
export LAMBDA_MEMORY_SIZE="3008"  # MB
export LAMBDA_TIMEOUT="300"       # seconds

# Application Configuration
export MIN_CLUSTER_SIZE="3"
export MAX_CLUSTERS="20"
```

### 4. Authenticate AWS CLI

Ensure your AWS CLI is configured:

```bash
aws configure
# OR
export AWS_PROFILE=your-profile-name
```

Verify authentication:

```bash
aws sts get-caller-identity
```

## Deployment Steps

Use the deployment script for fully automated deployment:

```bash
# Make script executable
chmod +x deploy.sh

# Load configuration and deploy
source deployment-config.sh && ./deploy.sh
```

The script will:
1. Create/verify ECR repository
2. Build Docker image (~5-10 minutes)
3. Push image to ECR
4. Deploy CloudFormation stack
5. Display API endpoint and key ID

Expected output:
```
========================================
Deployment Successful!
========================================

API Gateway URL:
  https://abc123xyz.execute-api.us-east-1.amazonaws.com/prod/analyze

API Key ID:
  def456uvw

To retrieve the actual API key value:
  aws apigateway get-api-key --api-key def456uvw --include-value --query value --output text
```

## Testing the Deployment

### 1. Retrieve API Key

```bash
# Get API Key ID from CloudFormation outputs
API_KEY_ID=$(aws cloudformation describe-stacks \
  --stack-name thematic-assessment-stack \
  --query 'Stacks[0].Outputs[?OutputKey==`ApiKeyId`].OutputValue' \
  --output text)

# Retrieve actual API key value
API_KEY=$(aws apigateway get-api-key \
  --api-key ${API_KEY_ID} \
  --include-value \
  --query value \
  --output text)

echo "Your API Key: ${API_KEY}"
```

### 2. Get API Endpoint

```bash
API_URL=$(aws cloudformation describe-stacks \
  --stack-name thematic-assessment-stack \
  --query 'Stacks[0].Outputs[?OutputKey==`ApiGatewayUrl`].OutputValue' \
  --output text)

echo "API URL: ${API_URL}"
```

### 3. Test Standalone Analysis

```bash
curl -X POST "${API_URL}" \
  -H "Content-Type: application/json" \
  -H "x-api-key: ${API_KEY}" \
  -d @test_request_standalone.json | jq
```

Expected response:
```json
{
  "clusters": [
    {
      "title": "Product Quality Praise",
      "sentiment": "positive",
      "sentences": ["r1", "r2", "r3", "r4", "r5", "r6"],
      "keyInsights": [
        "Customers consistently praise build quality and materials",
        "Product quality exceeds customer expectations"
      ]
    },
    {
      "title": "Pricing Concerns",
      "sentiment": "negative",
      "sentences": ["r7", "r8", "r9", "r10"],
      "keyInsights": [
        "Price point is a major concern for customers",
        "Product perceived as overpriced compared to competitors"
      ]
    }
  ]
}
```

### 4. Test Comparative Analysis

```bash
curl -X POST "${API_URL}" \
  -H "Content-Type: application/json" \
  -H "x-api-key: ${API_KEY}" \
  -d @test_request_comparative.json | jq
```

### 5. Test Error Handling

```bash
# Test invalid JSON
curl -X POST "${API_URL}" \
  -H "Content-Type: application/json" \
  -H "x-api-key: ${API_KEY}" \
  -d "invalid json" | jq

# Test missing API key
curl -X POST "${API_URL}" \
  -H "Content-Type: application/json" \
  -d @test_request_standalone.json | jq
```

## Monitoring and Troubleshooting

### View Lambda Logs

```bash
# Stream logs
aws logs tail /aws/lambda/thematic-assessment-function --follow

# Get recent errors
aws logs filter-log-events \
  --log-group-name /aws/lambda/thematic-assessment-function \
  --filter-pattern "ERROR" \
  --max-items 50
```

### CloudWatch Metrics

Navigate to CloudWatch Console:
- **Invocations**: Number of Lambda invocations
- **Duration**: Execution time (should be 10-60 seconds)
- **Errors**: Failed invocations
- **Throttles**: Rate limit hits
- **Memory Usage**: Should be < 3008 MB

### Common Issues

#### 1. "403 Forbidden" Error

**Cause**: Missing or invalid API key

**Solution**:
```bash
# Verify API key is correct
aws apigateway get-api-key --api-key YOUR_KEY_ID --include-value

# Ensure header is x-api-key (lowercase)
curl -H "x-api-key: YOUR_KEY" ...
```

#### 2. "500 Internal Server Error"

**Cause**: Lambda execution failure

**Solution**: Check CloudWatch logs
```bash
aws logs tail /aws/lambda/thematic-assessment-function --follow
```

Common causes:
- Secrets Manager permission issue
- OpenAI API key invalid
- Insufficient memory

#### 3. "Task timed out after 300.00 seconds"

**Cause**: Request processing took > 5 minutes

**Solution**: Increase timeout in CloudFormation template
```yaml
LambdaTimeout: 600  # 10 minutes
```

Redeploy:
```bash
source deployment-config.sh && ./deploy.sh
```

#### 4. Cold Start Delays

**First invocation**: 20-40 seconds (loading ML models)
**Warm invocations**: ~10 seconds

**Mitigation**:
- Use provisioned concurrency (adds cost)
- Accept cold starts for low-traffic APIs


## Cleanup

### Delete CloudFormation Stack

```bash
aws cloudformation delete-stack --stack-name thematic-assessment-stack

# Wait for deletion
aws cloudformation wait stack-delete-complete --stack-name thematic-assessment-stack
```

### Delete ECR Images

```bash
# List images
aws ecr list-images --repository-name thematic-assessment

# Delete all images
aws ecr batch-delete-image \
  --repository-name thematic-assessment \
  --image-ids imageTag=latest

# Delete repository
aws ecr delete-repository --repository-name thematic-assessment --force
```

### Delete Secrets Manager Secret

```bash
# Schedule deletion (7-30 day recovery window)
aws secretsmanager delete-secret \
  --secret-id thematic-assessment-openai-key \
  --recovery-window-in-days 7

# Force immediate deletion (NO RECOVERY)
aws secretsmanager delete-secret \
  --secret-id thematic-assessment-openai-key \
  --force-delete-without-recovery
```

## Cost Estimate

Estimated monthly cost for **1000 requests/month**:

| Service | Usage | Monthly Cost |
|---------|-------|--------------|
| Lambda | 1000 invocations × 30s × 3008MB | ~$0.10 |
| API Gateway | 1000 requests | ~$0.004 |
| Secrets Manager | 1 secret | $0.40 |
| ECR | 5 GB storage | $0.50 |
| CloudWatch Logs | 1 GB | $0.50 |
| **Total** | | **~$1.54/month** |

**OpenAI API costs** (not included above):
- GPT-4o: $2.50 per 1M input tokens, $10 per 1M output tokens
- Estimated $0.01-0.05 per analysis request
