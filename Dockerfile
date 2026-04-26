FROM python:3.10-slim

WORKDIR /app

# Install dependencies first (to cache this layer)
RUN apt-get update && apt-get install -y cmake build-essential git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install uv && uv pip install --system --no-cache -r requirements.txt

# Do not copy code here as it will be mounted at runtime via docker-compose
# Run Streamlit on port 8501 inside the container
CMD ["streamlit", "run", "BTX.py", "--server.port=8501", "--server.address=0.0.0.0"]
