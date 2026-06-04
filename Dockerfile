# memovox — container image (M3.3).
# Default: the free, stdlib-only REST server. Override CMD to run the worker or
# install the [serve] extra for FastAPI/uvicorn.
FROM python:3.12-slim

WORKDIR /app
COPY . /app

# Free path (no heavy ML deps). For the production server: pip install '.[serve]'.
RUN pip install --no-cache-dir .

# Store lives in a mounted volume so ingest/consolidate persist across restarts.
ENV MEMOVOX_STORE=/data
VOLUME ["/data"]
EXPOSE 8808

# Free stdlib server. Override for FastAPI:  CMD ["memovox","serve","--fastapi","--host","0.0.0.0"]
# Run a worker alongside the API (shared /data volume):  CMD ["memovox-worker"]
CMD ["memovox", "serve", "--host", "0.0.0.0"]
