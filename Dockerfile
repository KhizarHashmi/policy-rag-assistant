FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY data ./data
EXPOSE 8000
# Index is built on first start if empty (needs GOOGLE_API_KEY), then the API is served.
CMD ["sh", "-c", "python -c 'from policy_rag.pipeline import get_pipeline as g; p=g(); p.retriever.ids or p.ingest()' && uvicorn policy_rag.api:app --host 0.0.0.0 --port 8000"]
