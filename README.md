# FastAPI AI Backend

## Overview
A production-ready FastAPI backend integrating Google Vertex AI, Cerebras, Gemini, Exa, and OpenAI for chat, research, code generation, and more.

## Features
- Multi-backend AI API (Vertex AI, Cerebras, Gemini, Exa, OpenAI)
- Async, scalable, and containerized
- Extensible endpoints for research, chat, code, and more

## Requirements
- Python 3.11+
- Docker (for containerized deployment)

## Setup
1. Clone the repository
2. Copy `.env.example` to `.env` and fill in your secrets
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running Locally
```bash
uvicorn main:app --reload
```

## Running in Production (Docker)
```bash
docker build -t fastapi-ai-backend .
docker run --env-file .env -p 8080:8080 fastapi-ai-backend
```

## Testing
```bash
pytest
```

## Deployment
- Use the provided Dockerfile for cloud or on-prem deployment
- Expose port 8080

## License
MIT

## API Usage

- **Healthcheck**
  ```bash
  curl http://localhost:8080/health
  # {"status": "ok"}
  ```
- **Root API**
  ```bash
  curl http://localhost:8080/api
  ```
- **Simple Alpha Service**
  ```bash
  curl -X POST http://localhost:8080/api/alpha/simple/test \
    -H 'Content-Type: application/json' \
    -d '{"service_name": "test", "query": "hello", "details": ""}'
  ```

## Troubleshooting
- Ensure all environment variables are set in `.env`.
- Check logs for errors: `docker logs <container>` or server output.
- For CORS issues, update allowed origins in `main.py`.
- For dependency issues, rebuild the Docker image.

## Blue/Green Deployment (Kubernetes)
- Deploy a new version alongside the old one (e.g., `fastapi-ai-backend-green`).
- Switch the service selector to the new deployment after health checks pass.
- Optionally, use Ingress rules for canary traffic splitting.

## Centralized Logging
- Logs are in JSON format for easy ingestion by ELK, Loki, or cloud logging.
- Mount `/app/logs` as a persistent volume (see `k8s-deployment.yaml`).
- Forward logs to your logging backend (e.g., Filebeat, Fluentd, or cloud agent).

## Persistent Storage
- Use the provided PVC in `k8s-deployment.yaml` for logs or results.
- Adjust storage size and mount path as needed for your workload.