# Use a slim, official Python base image — small size, faster builds/pulls
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy only requirements first (Docker caches this layer separately —
# if your code changes but dependencies don't, rebuilds are much faster)
COPY requirements.txt .
# Install dependencies one at a time so a slow/failed download doesn't force
# re-downloading everything else — each successful line is cached separately.
RUN pip install --no-cache-dir --default-timeout=300 --retries=10 fastapi==0.115.5
RUN pip install --no-cache-dir --default-timeout=300 --retries=10 uvicorn==0.32.1
RUN pip install --no-cache-dir --default-timeout=300 --retries=10 pydantic==2.10.3
RUN pip install --no-cache-dir --default-timeout=300 --retries=10 numpy==2.0.2
RUN pip install --no-cache-dir --default-timeout=300 --retries=10 scipy
RUN pip install --no-cache-dir --default-timeout=300 --retries=10 scikit-learn==1.9.0
RUN pip install --no-cache-dir --default-timeout=300 --retries=10 pandas==2.2.3
RUN pip install --no-cache-dir --default-timeout=300 --retries=10 joblib==1.6.0

# Now copy the rest of the application code and model artifacts
COPY main.py .
COPY parkinsons_voice_model.pkl .
COPY scaler.pkl .

# The port uvicorn will listen on inside the container (Render/most hosts override this via $PORT)
EXPOSE 8000

# Run the API when the container starts.
# Uses shell form so $PORT (set by the hosting platform) is respected, falling back to 8000 locally.
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}