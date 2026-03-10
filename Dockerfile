# ==========================================
# ENTERPRISE DOCKERFILE (MULTI-STAGE)
# ==========================================

# Stage 1: Builder
FROM python:3.10-slim AS builder

WORKDIR /app

# Install system dependencies required for compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Production Runner
FROM python:3.10-slim

WORKDIR /app

# Create a non-root user for security compliance
RUN useradd -m -s /bin/bash appuser

# Copy installed dependencies from the builder stage
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

# Copy application source code
COPY ./src ./src
COPY .env.example .env

# Set ownership to the non-root user
RUN chown -R appuser:appuser /app
USER appuser

# Expose the API Gateway port
EXPOSE 8000

# Healthcheck for container orchestration (Kubernetes/Docker Swarm)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the FastAPI server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
