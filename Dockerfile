FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (to cache this layer)
RUN apt-get update && apt-get install -y cmake build-essential git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install uv && uv pip install --system --no-cache -r requirements.txt

# Do not copy code here as it will be mounted at runtime via docker-compose
# Default CMD is overridden by docker-compose; batch uses 8503, single-image uses 8504.
CMD ["streamlit", "run", "BTX.py", "--server.port=8504", "--server.address=0.0.0.0"]
