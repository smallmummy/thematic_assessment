#!/bin/bash
# Example deployment configuration
# Copy this file to deployment-config.sh and modify with your values
# Then source it before running deploy.sh: source deployment-config.sh && ./deploy.sh

# AWS Configuration
export AWS_REGION="us-east-1"
export AWS_ACCOUNT_ID="123456789012"  # Your AWS account ID (or leave empty to auto-detect)

# Project Configuration
export PROJECT_NAME="thematic-assessment"
export ECR_REPO_NAME="thematic-assessment"
export IMAGE_TAG="latest"
export STACK_NAME="thematic-assessment-stack"

# Secrets Manager
# IMPORTANT: Create this secret first using:
# aws secretsmanager create-secret \
#   --name thematic-assessment-openai-key \
#   --description "OpenAI API key for thematic assessment service" \
#   --secret-string "sk-your-actual-openai-api-key-here"
export OPENAI_API_KEY_SECRET_ARN="arn:aws:secretsmanager:us-east-1:123456789012:secret:thematic-assessment-openai-key-XXXXXX"

# Lambda Configuration
export LAMBDA_MEMORY_SIZE="3008"  # MB (2048-10240)
export LAMBDA_TIMEOUT="300"       # seconds (60-900)

# Application Configuration
export MIN_CLUSTER_SIZE="3"       # HDBSCAN min cluster size (2-10)
export MAX_CLUSTERS="20"          # Max clusters to return (5-50)

echo "Deployment configuration loaded"
echo "  Region: ${AWS_REGION}"
echo "  Project: ${PROJECT_NAME}"
echo "  Stack: ${STACK_NAME}"