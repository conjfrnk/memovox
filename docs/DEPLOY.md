# Deploying memovox

memovox runs in three modes, all sharing one SQLite store directory.

## 1. Free (stdlib, default)

No third-party server. The standard-library `http.server` REST API plus inline /
in-process jobs:

```bash
pip install .
memovox serve --host 0.0.0.0 --port 8808     # REST API
memovox mcp                                   # MCP server (stdio JSON-RPC)
```

Long operations (`consolidate`) enqueue a background job and return a `job_id`
immediately; a single-process server auto-spawns one in-process worker thread to
drain its own queue.

## 2. Serve (FastAPI/uvicorn, `[serve]` extra)

For a production HTTP server, install the optional extra and pass `--fastapi`:

```bash
pip install '.[serve]'
memovox serve --fastapi --host 0.0.0.0 --port 8808
```

The FastAPI app mounts the **same** route functions (`server/routes.py`) as the
stdlib server — a parity test guarantees byte-identical JSON. FastAPI is never
imported on the free path.

## 3. Worker (separate process)

For a deployed setup, run the worker next to the API so the API process never
blocks on jobs. Both mount the **same store directory** (SQLite WAL + `busy_timeout`
make this safe at the default `--concurrency 1`):

```bash
memovox-worker                 # poll forever
memovox-worker --once          # drain the queue then exit (cron-friendly)
memovox worker --once          # same, via the main CLI
```

The job runner is `queued → running → succeeded|failed` with attempt counting,
exponential-backoff retry, idempotent `(kind, args_hash)` de-dup, and crash
resumability (a job left `running` by a crashed worker is re-claimed on restart).

> **Throughput honesty.** `--concurrency 1` is the only deterministic, gated mode;
> it is *not* faster, just off the caller thread. `--concurrency > 1` is opt-in and
> **out of the eval gate**. The real throughput bottleneck is the visual track, not
> job scheduling — frame parallelism is a separate effort and this worker does not
> claim a speedup.

## Docker

```bash
docker build -t memovox .
docker run -v $(pwd)/data:/data -p 8808:8808 memovox          # free REST server
docker run -v $(pwd)/data:/data memovox memovox-worker        # worker, shared store
```

## Privacy & terms (private-by-default posture)

memovox is **private by default**: the store is a local directory you own; nothing
is uploaded anywhere. When ingesting other people's media:

- **Respect each source's Terms of Service.** URL ingestion uses `yt-dlp`; you are
  responsible for your right to download and index a given source.
- **Air-gapped / no-egress mode.** Set `local_only=true` (Settings /
  `MEMOVOX_LOCAL_ONLY=1`) to refuse all network acquisition — URL ingests raise a
  clear error before any fetch, and only local files are accepted.
- **Retention & redaction.** Delete a video and all its derived moments/claims/edges
  with `mv.delete_video(video_id)` (`memovox forget <video_id>`) — the redaction
  primitive. Nothing else is retained outside the store directory.
