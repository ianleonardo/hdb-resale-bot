# Build from repository root:
#   docker build -f backend/Dockerfile -t hdb-backend .
FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/app ./app
COPY data/hdb_rpi.csv ./data/hdb_rpi.csv
ENV PYTHONUNBUFFERED=1
ENV HDB_RPI_PATH=/app/data/hdb_rpi.csv
EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
