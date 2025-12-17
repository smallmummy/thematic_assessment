# Step-by-step commands to rebuild and redeploy with permissions fix

# 1. Load your deployment configuration
source ../deploy/deployment-config.sh

# 2. Rebuild the Docker image with permission fix
docker buildx build --platform linux/amd64 --provenance=false -t thematic-assessment:latest -f ../Dockerfile ..

# 3. Set ECR URI
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/thematic-assessment"

# 4. Authenticate Docker to ECR
aws ecr get-login-password --region ${AWS_REGION} | \
  docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# 5. Tag the new image
docker tag thematic-assessment:latest ${ECR_URI}:latest

# 6. Push to ECR
docker push ${ECR_URI}:latest

# 7. Update Lambda function with new image
aws lambda update-function-code \
  --function-name thematic-assessment-function \
  --image-uri ${ECR_URI}:latest \
  --region ${AWS_REGION}

# 8. Wait for update to complete (optional but recommended)
aws lambda wait function-updated \
  --function-name thematic-assessment-function \
  --region ${AWS_REGION}

echo "Deployment complete! You can now test your API."
