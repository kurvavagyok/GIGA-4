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