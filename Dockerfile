# NMJ analysis container — Python 3.12.13, package pins in requirements.txt
FROM python:3.12.13-slim

WORKDIR /app

# Build tools for native wheels (e.g. aicspylibczi)
RUN apt-get update && apt-get install -y cmake build-essential git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir uv==0.9.7 \
    && uv pip install --system --no-cache -r requirements.txt

# Code is mounted at runtime via docker-compose
# Default CMD is overridden by docker-compose; batch uses 8503, single-image uses 8504.
CMD ["streamlit", "run", "BTX.py", "--server.port=8504", "--server.address=0.0.0.0"]
