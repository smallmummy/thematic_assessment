# Multi-stage Dockerfile for AWS Lambda Container Image
# Optimized for ML workloads with sentence-transformers and transformers

FROM public.ecr.aws/lambda/python:3.12 AS builder

# Install system dependencies for building Python packages
RUN dnf install gcc -y

# # Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --target "${LAMBDA_TASK_ROOT}"


# Final stage - minimal runtime image
FROM public.ecr.aws/lambda/python:3.12

# Copy installed packages from builder
COPY --from=builder ${LAMBDA_TASK_ROOT} ${LAMBDA_TASK_ROOT}

# Copy application code
COPY lambda_handler.py ${LAMBDA_TASK_ROOT}/
COPY src/ ${LAMBDA_TASK_ROOT}/src/

# Fix permissions for Lambda runtime
RUN chmod 644 ${LAMBDA_TASK_ROOT}/lambda_handler.py
RUN chmod -R 755 ${LAMBDA_TASK_ROOT}/src/

# Set cache directories for model downloads (must match runtime settings)
# Models will be cached in the image at these locations
ENV TRANSFORMERS_CACHE=${LAMBDA_TASK_ROOT}/.cache/transformers
ENV HF_HOME=${LAMBDA_TASK_ROOT}/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=${LAMBDA_TASK_ROOT}/.cache/sentence_transformers
ENV TORCH_HOME=${LAMBDA_TASK_ROOT}/.cache/torch

# Pre-download ML models to reduce cold start time
# Models are cached in the image at the locations specified above
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='finiteautomata/bertweet-base-sentiment-analysis')"

# Set permissions for cached models
RUN chmod -R 755 ${LAMBDA_TASK_ROOT}/.cache

# Set working directory
WORKDIR ${LAMBDA_TASK_ROOT}

# Set the Lambda handler
CMD ["lambda_handler.lambda_handler"]