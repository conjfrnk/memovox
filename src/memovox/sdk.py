"""The Python SDK facade (spec §8).

    from memovox import Memovox
    mv = Memovox(store="~/knowledge")
    mv.ingest("https://youtu.be/...")
    ans = mv.ask("what's the recommended chunk size, and who recommended it?")
    for c in ans.citations:
        print(c.video_id, c.t_start_s, c.deep_link)
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass, field
from typing import List, Optional

from . import augur, pipeline
from .backends import backend_status, get_embedder, get_llm, get_nli, get_reranker
from .config import Config, Settings
from .loom import LoomStore, Video
from .loom.consolidate import (
    ContradictionPair,
    consolidate as run_consolidation,
    find_contradictions,
)
from .loom.digest import build_digest
from .loom.evolution import claim_evolution


@dataclass
class SyncReport:
    """Result of one ``Memovox.sync()`` (M3.2): per-entry status + batch counts."""

    entries: List[dict] = field(default_factory=list)
    n_new: int = 0
    n_skipped: int = 0
    n_failed: int = 0

    def to_dict(self) -> dict:
        return {"n_new": self.n_new, "n_skipped": self.n_skipped,
                "n_failed": self.n_failed, "entries": self.entries}


class Memovox:
    def __init__(self, store: Optional[str] = None, settings: Optional[Settings] = None, **overrides):
        # store=None (not "~/.memovox") so Config's resolution order holds:
        # explicit arg > $MEMOVOX_STORE > ~/.memovox. A truthy literal default here
        # shadowed the env lookup, so Memovox() ignored $MEMOVOX_STORE (the CLI
        # passes None and worked). See tests/test_integration.TestSdkStoreResolution.
        self.config = Config(store=store, settings=settings)
        if overrides:
            self.config.settings = self.config.settings.merged(overrides)
        self._worker = None
        self._worker_lock = threading.Lock()

    def close(self) -> None:
        """Stop the auto-spawned background worker (if any) and release its resources.
        Idempotent; safe to call even if no worker was spawned."""
        worker = self._worker
        if worker is not None:
            worker.stop()
            worker.join(timeout=5.0)
            self._worker = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def settings(self) -> Settings:
        return self.config.settings

    # -- ingestion ---------------------------------------------------------

    def ingest(self, source: str, **kwargs) -> pipeline.IngestReport:
        return pipeline.ingest(self.config, source, settings=self.settings, **kwargs)

    def sync(self) -> "SyncReport":
        """Subscription sync (M3.2): for each subscribed source, enumerate its
        videos (metadata only), skip ids already in the cursor, ingest the rest with
        the whole-corpus resolve DEFERRED, isolate per-entry failures, then run ONE
        corpus resolve + one incremental consolidation for the batch. Idempotent: a
        re-sync with no new uploads ingests nothing. All diagnostics go to stderr."""
        from . import pipeline, sync_state
        from .stentor import enumerate_source

        entries: List[dict] = []
        n_new = n_skipped = n_failed = 0
        any_new = False
        with LoomStore(self.config) as store:
            for src in self._load_sources():
                # M3.3: local_only refuses remote enumeration too (not just ingest),
                # so NO network egress happens for a URL source when private.
                if self.settings.local_only and src.startswith(("http://", "https://")):
                    print(f"sync: local_only set, skipping remote source {src}", file=sys.stderr)
                    entries.append({"url": src, "status": "skipped_local_only"})
                    continue
                try:
                    enumerated = enumerate_source(self.config, src)
                except Exception as exc:  # enumeration failure is per-source, not fatal
                    print(f"sync: enumerate failed for {src}: {exc}", file=sys.stderr)
                    entries.append({"url": src, "status": "enumerate_failed", "error": str(exc)[:200]})
                    continue
                seen = sync_state.seen_ids(store, src)
                for e in enumerated:
                    if e.video_id in seen:
                        n_skipped += 1
                        entries.append({"video_id": e.video_id, "url": e.url, "status": "skipped"})
                        continue
                    try:
                        pipeline.ingest(self.config, e.url, settings=self.settings,
                                        store=store, resolve_corpus=False)
                    except Exception as exc:  # one bad entry never aborts the batch
                        print(f"sync: ingest failed for {e.url}: {exc}", file=sys.stderr)
                        n_failed += 1
                        entries.append({"video_id": e.video_id, "url": e.url,
                                        "status": "failed", "error": str(exc)[:200]})
                        continue
                    sync_state.mark_seen(store, src, e.video_id)  # only on success
                    n_new += 1
                    any_new = True
                    entries.append({"video_id": e.video_id, "url": e.url, "status": "new"})
            if any_new:
                pipeline.resolve_corpus_pass(self.config, store, self.settings)
        if any_new:
            self.consolidate()  # one incremental (M0.2 watermark) consolidation per batch
        return SyncReport(entries=entries, n_new=n_new, n_skipped=n_skipped, n_failed=n_failed)

    def _load_sources(self) -> List[str]:
        import json

        path = self.config.subscriptions_path
        if not path.exists():
            return []
        try:
            subs = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            print(f"memovox: ignoring malformed {path} ({exc}).", file=sys.stderr)
            return []
        if not isinstance(subs, dict):
            print(f"memovox: {path} is not a JSON object; ignoring.", file=sys.stderr)
            return []
        sources = subs.get("sources")
        out = []
        for entry in sources if isinstance(sources, list) else []:
            url = entry.get("url") if isinstance(entry, dict) else entry
            if url and isinstance(url, str):
                out.append(url)
        return out

    def _write_sources(self, sources: List[str]) -> None:
        import json

        path = self.config.subscriptions_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"sources": [{"url": u} for u in sources]}, indent=2),
                        encoding="utf-8")

    def subscribe(self, url: str) -> List[str]:
        """Add a channel/playlist/video URL to subscriptions.json (idempotent)."""
        sources = self._load_sources()
        if url not in sources:
            sources.append(url)
            self._write_sources(sources)
        return sources

    def unsubscribe(self, url: str) -> List[str]:
        """Remove a source from subscriptions.json (no-op if absent)."""
        sources = [s for s in self._load_sources() if s != url]
        self._write_sources(sources)
        return sources

    def list_subscriptions(self) -> List[str]:
        return self._load_sources()

    # -- query -------------------------------------------------------------

    def ask(self, query: str, *, video_id: Optional[str] = None,
            modality: str = "any") -> augur.Answer:
        with LoomStore(self.config) as store:
            embedder = get_embedder(self.settings.embed_backend, config=self.config)
            llm = get_llm(self.settings.llm_backend, config=self.config)
            reranker = get_reranker(self.settings.rerank_backend, config=self.config)
            return augur.ask(
                store, query, embedder=embedder, llm=llm, settings=self.settings,
                video_id=video_id, modality=modality, reranker=reranker,
            )

    def contradictions(self, topic: Optional[str] = None) -> List[ContradictionPair]:
        with LoomStore(self.config) as store:
            nli = get_nli(self.settings.nli_backend, config=self.config)
            return find_contradictions(
                store, nli=nli, topic=topic, threshold=self.settings.contradiction_threshold
            )

    def consolidate(self) -> dict:
        """Run the cross-corpus consolidation background job (spec §4 stage 7):
        topic induction, contradiction/agreement detection, consensus clustering,
        and dedup. Run after ingesting new videos; returns a counts report.

        SYNCHRONOUS and unchanged — the eval harness + incremental_equivalence gate
        depend on this inline path. ``enqueue_consolidate`` is the async alternative."""
        with LoomStore(self.config) as store:
            nli = get_nli(self.settings.nli_backend, config=self.config)
            return run_consolidation(store, nli=nli, settings=self.settings).to_dict()

    # -- async jobs (M3.3) -------------------------------------------------

    def _jobstore(self):
        from .serving.jobs import JobStore
        return JobStore(self.config)

    def _ensure_worker(self) -> None:
        """Auto-spawn ONE daemon JobWorker per process on first enqueue, so a
        single-process MCP/SDK caller drains its own queue without a separate
        memovox-worker. A deployed setup runs memovox-worker explicitly instead."""
        from .serving.jobs import JobWorker
        with self._worker_lock:  # guard the check-then-act so we never spawn duplicates
            worker = self._worker
            if worker is None or not worker.is_alive():
                self._worker = JobWorker(self)
                self._worker.start()

    def enqueue_consolidate(self) -> dict:
        """Enqueue an async consolidation; returns {job_id, state} immediately."""
        with self._jobstore() as jobs:
            job_id = jobs.enqueue("consolidate", {})
            state = jobs.get_job(job_id)["state"]
        self._ensure_worker()
        return {"job_id": job_id, "state": state}

    def enqueue_ingest(self, source: str, *, source_url: Optional[str] = None,
                       title: Optional[str] = None) -> dict:
        """Enqueue an async ingest; returns {job_id, state} immediately. A real
        ingest (download + ASR + NLI) runs minutes — longer than MCP clients wait
        (Claude Desktop cancels at 240 s) — so callers poll job_status instead.
        None-valued args are dropped to keep the (kind, args_hash) de-dup stable."""
        args = {"source": source}
        if source_url:
            args["source_url"] = source_url
        if title:
            args["title"] = title
        with self._jobstore() as jobs:
            job_id = jobs.enqueue("ingest", args)
            state = jobs.get_job(job_id)["state"]
        self._ensure_worker()
        return {"job_id": job_id, "state": state}

    def enqueue_sync(self) -> dict:
        with self._jobstore() as jobs:
            job_id = jobs.enqueue("sync", {})
            state = jobs.get_job(job_id)["state"]
        self._ensure_worker()
        return {"job_id": job_id, "state": state}

    def job_status(self, job_id: str) -> Optional[dict]:
        """Resolve a job id to {state, result, error, attempts} (None if unknown)."""
        import json

        with self._jobstore() as jobs:
            job = jobs.get_job(job_id)
        if job is None:
            return None
        return {
            "job_id": job["job_id"], "kind": job["kind"], "state": job["state"],
            "attempts": job["attempts"], "error": job["error"],
            "result": json.loads(job["result_json"]) if job["result_json"] else None,
        }

    def synthesize(self, topic: str) -> augur.Synthesis:
        """Corpus-level "literature review" of what the sources say about a topic
        (consensus + disagreements), grounded and cited (spec §5)."""
        with LoomStore(self.config) as store:
            nli = get_nli(self.settings.nli_backend, config=self.config)
            llm = get_llm(self.settings.llm_backend, config=self.config)
            embedder = get_embedder(self.settings.embed_backend, config=self.config)
            return augur.synthesize(store, topic, nli=nli, llm=llm, embedder=embedder,
                                    settings=self.settings)

    def evolution(self, *, entity: Optional[str] = None, topic: Optional[str] = None) -> List[dict]:
        """Trace how a claim/position about an entity or topic changed over time.

        Returns ordered evolution steps (oldest source first), each with its
        relation to the previous step (CONTRADICTS/CORRECTS/SUPPORTS) and a deep
        link (spec §5).
        """
        with LoomStore(self.config) as store:
            steps = claim_evolution(store, entity_id=entity, topic=topic)
            return [s.to_dict() for s in steps]

    def claim_history(self, claim_id: str) -> List[dict]:
        """Every version in a claim's supersede lineage, oldest→newest (M3.1, §2).
        Nothing is deleted — superseded versions are returned alongside the live one."""
        with LoomStore(self.config) as store:
            return [c.to_dict() for c in store.claim_history(claim_id)]

    # -- read / export -----------------------------------------------------

    def export(self, video_id: str, fmt: str = "md") -> str:
        with LoomStore(self.config) as store:
            if fmt == "md":
                return build_digest(store, video_id)
            if fmt == "json":
                import json

                video = store.get_video(video_id)
                if not video:
                    raise KeyError(f"No video {video_id!r}")
                moments = store.moments_for_video(video_id)
                claims = store.claims_for_video(video_id, status=None)
                return json.dumps(
                    {
                        "video": video.to_dict(),
                        "moments": [m.to_dict() for m in moments],
                        "claims": [c.to_dict() for c in claims],
                    },
                    indent=2, ensure_ascii=False,
                )
            raise ValueError(f"Unknown export format {fmt!r} (use 'md' or 'json').")

    def list_videos(self) -> List[Video]:
        with LoomStore(self.config) as store:
            return store.list_videos()

    def delete_video(self, video_id: str) -> bool:
        """Redaction primitive (M3.3/§12): delete a video and all its derived data — the
        SQLite store AND the on-disk human-readable digest. Returns True if it existed."""
        from .util import digest_filename, slugify
        with LoomStore(self.config) as store:
            deleted = store.delete_video(video_id)
        if deleted:
            # The digest .md lives on the FILESYSTEM, outside the SQLite store the store-layer
            # delete scrubs — yet it holds claim text, resolved speaker names, OCR'd on-screen
            # text and source deep links (the most exposed surface, a plaintext file a user may
            # sync to cloud / commit to git). Remove it so "forget this video" really forgets —
            # both the injective name and the legacy slug-only name (pre-injective-scheme stores).
            for fname in (digest_filename(video_id), f"{slugify(video_id)}.md"):
                (self.config.digests_dir / fname).unlink(missing_ok=True)
        return deleted

    def get_provenance(self, claim_id: str) -> Optional[dict]:
        from .loom.models import STATUS_COMMITTED, make_provenance

        with LoomStore(self.config) as store:
            claim = store.get_claim(claim_id)
            if not claim:
                return None
            # Only a COMMITTED claim (one that passed the entailment gate) carries trusted
            # provenance. An unsupported/superseded claim is retained for the audit trail,
            # but every other user-facing surface excludes it — so do NOT hand back a
            # verified-looking deep link for one. Withhold provenance and flag trusted=False
            # so a client can't mistake a rejected claim's citation for a vetted one.
            trusted = claim.status == STATUS_COMMITTED
            video = store.get_video(claim.video_id) if trusted else None
            prov = make_provenance(
                video, claim.t_start_s, claim.t_end_s, speaker=claim.speaker_id,
                confidence=claim.entailment_score,
            ) if video else None
            return {
                "claim": claim.to_dict(),
                "provenance": prov.to_dict() if prov else None,
                "trusted": trusted,
            }

    def extract(self, video_id: str, *, use_llm: bool = False) -> dict:
        """Structured-JSON extraction document for a video (spec §5, X.4):
        typed claims + resolved entity surfaces, schema-validated, deterministic
        on the free (rule-based) path. Distinct from ask()/synthesize() answers."""
        from .assay.extract_output import extract_video_document

        with LoomStore(self.config) as store:
            moments = store.moments_for_video(video_id)
            llm = get_llm(self.settings.llm_backend, config=self.config) if use_llm else None
            return extract_video_document(moments, llm=llm)

    def stats(self) -> dict:
        with LoomStore(self.config) as store:
            return store.stats()

    def metrics(self, *, video_id: Optional[str] = None) -> dict:
        """Per-video stage metrics + the cumulative ledger (M0.1 observability)."""
        with LoomStore(self.config) as store:
            ledger = store.metrics_ledger()
            if video_id:
                stage = {video_id: store.stage_metrics(video_id)}
            else:
                stage = {v.video_id: store.stage_metrics(v.video_id) for v in store.list_videos()}
            return {"ledger": ledger, "stage_metrics": stage}

    def backends(self) -> dict:
        return backend_status()
