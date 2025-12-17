#!/bin/bash
set -e

# Deployment script for Thematic Assessment Microservice
# This script builds the Docker image, pushes to ECR, and deploys via CloudFormation

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration (these should be set as environment variables or modified here)
AWS_REGION="${AWS_REGION}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
PROJECT_NAME="${PROJECT_NAME:-thematic-assessment}"
ECR_REPO_NAME="${ECR_REPO_NAME:-${PROJECT_NAME}}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
STACK_NAME="${STACK_NAME:-${PROJECT_NAME}-stack}"

# CloudFormation parameters (modify as needed)
OPENAI_API_KEY_SECRET_ARN="${OPENAI_API_KEY_SECRET_ARN}"
LAMBDA_MEMORY_SIZE="${LAMBDA_MEMORY_SIZE:-3008}"
LAMBDA_TIMEOUT="${LAMBDA_TIMEOUT:-300}"
MIN_CLUSTER_SIZE="${MIN_CLUSTER_SIZE:-3}"
MAX_CLUSTERS="${MAX_CLUSTERS:-20}"

# Derived variables
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}"
IMAGE_URI="${ECR_URI}:${IMAGE_TAG}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Thematic Assessment Deployment Script${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Configuration:"
echo "  AWS Region: ${AWS_REGION}"
echo "  AWS Account: ${AWS_ACCOUNT_ID}"
echo "  Project Name: ${PROJECT_NAME}"
echo "  ECR Repository: ${ECR_REPO_NAME}"
echo "  Image Tag: ${IMAGE_TAG}"
echo "  Stack Name: ${STACK_NAME}"
echo ""

# Validation
if [ -z "$OPENAI_API_KEY_SECRET_ARN" ]; then
    echo -e "${RED}Error: OPENAI_API_KEY_SECRET_ARN environment variable not set${NC}"
    echo "Please set it to your Secrets Manager secret ARN:"
    echo "  export OPENAI_API_KEY_SECRET_ARN=arn:aws:secretsmanager:REGION:ACCOUNT:secret:NAME"
    exit 1
fi

# Step 1: Check if ECR repository exists, create if not
echo -e "${YELLOW}[1/6] Checking ECR repository...${NC}"
if ! aws ecr describe-repositories --repository-names "${ECR_REPO_NAME}" --region "${AWS_REGION}" &> /dev/null; then
    echo "  Creating ECR repository: ${ECR_REPO_NAME}"
    aws ecr create-repository \
        --repository-name "${ECR_REPO_NAME}" \
        --region "${AWS_REGION}" \
        --image-scanning-configuration scanOnPush=true \
        --encryption-configuration encryptionType=AES256
    echo -e "${GREEN}  ✓ ECR repository created${NC}"
else
    echo -e "${GREEN}  ✓ ECR repository exists${NC}"
fi

# Step 2: Authenticate Docker to ECR
echo -e "${YELLOW}[2/6] Authenticating Docker to ECR...${NC}"
aws ecr get-login-password --region "${AWS_REGION}" | \
    docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
echo -e "${GREEN}  ✓ Docker authenticated${NC}"

# Step 3: Build Docker image
echo -e "${YELLOW}[3/6] Building Docker image...${NC}"
echo "  This may take 5-10 minutes (downloading ML models)"
docker buildx build --platform linux/amd64 --provenance=false -t "${PROJECT_NAME}:${IMAGE_TAG}" -f ../Dockerfile ..
echo -e "${GREEN}  ✓ Docker image built${NC}"

# Step 4: Tag and push to ECR
echo -e "${YELLOW}[4/6] Pushing image to ECR...${NC}"
docker tag "${PROJECT_NAME}:${IMAGE_TAG}" "${IMAGE_URI}"
docker push "${IMAGE_URI}"
echo -e "${GREEN}  ✓ Image pushed to ECR${NC}"

# Step 5: Deploy CloudFormation stack
echo -e "${YELLOW}[5/6] Deploying CloudFormation stack...${NC}"
aws cloudformation deploy \
    --template-file ../cloudformation/cloudformation.yaml \
    --stack-name "${STACK_NAME}" \
    --region "${AWS_REGION}" \
    --parameter-overrides \
        ProjectName="${PROJECT_NAME}" \
        OpenAIApiKeySecretArn="${OPENAI_API_KEY_SECRET_ARN}" \
        LambdaMemorySize="${LAMBDA_MEMORY_SIZE}" \
        LambdaTimeout="${LAMBDA_TIMEOUT}" \
        MinClusterSize="${MIN_CLUSTER_SIZE}" \
        MaxClusters="${MAX_CLUSTERS}" \
        EcrRepositoryUri="${ECR_URI}" \
        ImageTag="${IMAGE_TAG}" \
    --capabilities CAPABILITY_NAMED_IAM \
    --no-fail-on-empty-changeset

echo -e "${GREEN}  ✓ CloudFormation stack deployed${NC}"

# Step 6: Retrieve outputs
echo -e "${YELLOW}[6/6] Retrieving deployment outputs...${NC}"
API_URL=$(aws cloudformation describe-stacks \
    --stack-name "${STACK_NAME}" \
    --region "${AWS_REGION}" \
    --query 'Stacks[0].Outputs[?OutputKey==`ApiGatewayUrl`].OutputValue' \
    --output text)

API_KEY_ID=$(aws cloudformation describe-stacks \
    --stack-name "${STACK_NAME}" \
    --region "${AWS_REGION}" \
    --query 'Stacks[0].Outputs[?OutputKey==`ApiKeyId`].OutputValue' \
    --output text)

LAMBDA_ARN=$(aws cloudformation describe-stacks \
    --stack-name "${STACK_NAME}" \
    --region "${AWS_REGION}" \
    --query 'Stacks[0].Outputs[?OutputKey==`LambdaFunctionArn`].OutputValue' \
    --output text)

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Deployment Successful!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "API Gateway URL:"
echo "  ${API_URL}"
echo ""
echo "API Key ID:"
echo "  ${API_KEY_ID}"
echo ""
echo "To retrieve the actual API key value:"
echo "  aws apigateway get-api-key --api-key ${API_KEY_ID} --include-value --query value --output text"
echo ""
echo "Lambda Function ARN:"
echo "  ${LAMBDA_ARN}"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "1. Retrieve your API key using the command above"
echo "2. Test the API:"
echo "   curl -X POST '${API_URL}' \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -H 'x-api-key: YOUR_API_KEY' \\"
echo "     -d @test_request.json"
echo ""