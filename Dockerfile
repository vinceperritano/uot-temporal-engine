FROM python:3.11-slim

WORKDIR /app

# Don't write .pyc files; log immediately
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY uot_api_v2.py          .
COPY uot_engine_v12_patched.py .
COPY uot_live_search.py     .
COPY uot_frontend_v2.html   .

# Data directory for persistent SQLite (mount a volume here on Railway/Render)
RUN mkdir -p /data
ENV UOT_DB_PATH=/data/uot_runs.db

# Port — Railway/Render inject $PORT automatically
EXPOSE 8000

# Start FastAPI
CMD ["sh", "-c", "uvicorn uot_api_v2:app --host 0.0.0.0 --port ${PORT:-8000}"]
