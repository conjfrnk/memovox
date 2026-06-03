"""The Python SDK facade (spec §8).

    from memovox import Memovox
    mv = Memovox(store="~/knowledge")
    mv.ingest("https://youtu.be/...")
    ans = mv.ask("what's the recommended chunk size, and who recommended it?")
    for c in ans.citations:
        print(c.video_id, c.t_start_s, c.deep_link)
"""

from __future__ import annotations

from typing import List, Optional

from . import augur, pipeline
from .backends import backend_status, get_embedder, get_llm, get_nli
from .config import Config, Settings
from .loom import LoomStore, Video
from .loom.consolidate import ContradictionPair, find_contradictions
from .loom.digest import build_digest
from .loom.evolution import claim_evolution


class Memovox:
    def __init__(self, store: str = "~/.memovox", settings: Optional[Settings] = None, **overrides):
        self.config = Config(store=store, settings=settings)
        if overrides:
            self.config.settings = self.config.settings.merged(overrides)

    @property
    def settings(self) -> Settings:
        return self.config.settings

    # -- ingestion ---------------------------------------------------------

    def ingest(self, source: str, **kwargs) -> pipeline.IngestReport:
        return pipeline.ingest(self.config, source, settings=self.settings, **kwargs)

    def sync(self) -> List[pipeline.IngestReport]:
        """Ingest new items from subscriptions.json (channels/playlists)."""
        import json

        path = self.config.subscriptions_path
        if not path.exists():
            return []
        try:
            subs = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return []
        reports = []
        for entry in subs.get("sources", []):
            url = entry.get("url") if isinstance(entry, dict) else entry
            if url:
                reports.append(self.ingest(url))
        return reports

    # -- query -------------------------------------------------------------

    def ask(self, query: str, *, video_id: Optional[str] = None) -> augur.Answer:
        with LoomStore(self.config) as store:
            embedder = get_embedder(self.settings.embed_backend, config=self.config)
            llm = get_llm(self.settings.llm_backend, config=self.config)
            return augur.ask(
                store, query, embedder=embedder, llm=llm, settings=self.settings, video_id=video_id
            )

    def contradictions(self, topic: Optional[str] = None) -> List[ContradictionPair]:
        with LoomStore(self.config) as store:
            nli = get_nli(self.settings.nli_backend, config=self.config)
            return find_contradictions(
                store, nli=nli, topic=topic, threshold=self.settings.contradiction_threshold
            )

    def synthesize(self, topic: str) -> augur.Synthesis:
        """Corpus-level "literature review" of what the sources say about a topic
        (consensus + disagreements), grounded and cited (spec §5)."""
        with LoomStore(self.config) as store:
            nli = get_nli(self.settings.nli_backend, config=self.config)
            llm = get_llm(self.settings.llm_backend, config=self.config)
            return augur.synthesize(store, topic, nli=nli, llm=llm, settings=self.settings)

    def evolution(self, *, entity: Optional[str] = None, topic: Optional[str] = None) -> List[dict]:
        """Trace how a claim/position about an entity or topic changed over time.

        Returns ordered evolution steps (oldest source first), each with its
        relation to the previous step (CONTRADICTS/CORRECTS/SUPPORTS) and a deep
        link (spec §5).
        """
        with LoomStore(self.config) as store:
            steps = claim_evolution(store, entity_id=entity, topic=topic)
            return [s.to_dict() for s in steps]

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

    def get_provenance(self, claim_id: str) -> Optional[dict]:
        with LoomStore(self.config) as store:
            claim = store.get_claim(claim_id)
            if not claim:
                return None
            video = store.get_video(claim.video_id)
            from .loom.models import make_provenance

            prov = make_provenance(
                video, claim.t_start_s, claim.t_end_s, speaker=claim.speaker_id,
                confidence=claim.entailment_score,
            ) if video else None
            return {
                "claim": claim.to_dict(),
                "provenance": prov.to_dict() if prov else None,
            }

    def stats(self) -> dict:
        with LoomStore(self.config) as store:
            return store.stats()

    def backends(self) -> dict:
        return backend_status()
