# Build from repository root:
#   docker build -f backend/Dockerfile -t hdb-backend .
# RPI series is loaded at runtime from GCS (same as model/metrics); see model_loader.RPI_BLOB.
FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Top-level module required to unpickle training/v4 ARIMABundle (pickle refs "arima_v4")
COPY training/v4/arima_v4.py ./arima_v4.py
COPY backend/app ./app
ENV PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
