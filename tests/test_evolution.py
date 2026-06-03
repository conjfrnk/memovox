"""W3 — first-class claim-evolution tracking (Phase 3, spec §5).

Order an entity's or topic's claims by source publish date to trace how a
position changes over time, annotating each step's relation (CONTRADICTS /
CORRECTS / SUPPORTS) to its predecessor by reading the persisted graph edges,
and flagging superseded claims. Undated sources sort last (matching Augur's
temporal answer ordering).
"""

import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox.backends.embed import HashingEmbedder
from memovox.config import Config
from memovox.loom import Claim, Entity, LoomStore, Moment, Video
from memovox.loom.evolution import claim_evolution


class EvolutionTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.config = Config(store=pathlib.Path(self._tmp.name) / "store").ensure()
        self.store = LoomStore(self.config)
        self.emb = HashingEmbedder(dim=256)

    def _video(self, vid, pub):
        self.store.upsert_video(Video(video_id=vid, source_url=f"https://x/{vid}",
                                      title=vid, content_hash=vid, published_at=pub))

    def _claim(self, cid, vid, text, *, status="committed", entity=None):
        mid = f"{vid}#m0"
        if self.store.get_moment(mid) is None:
            self.store.add_moment(Moment(mid, vid, 0.0, 10.0, text, "spk_0", index=0),
                                  self.emb.embed_one(text))
        c = Claim(claim_id=cid, moment_id=mid, video_id=vid, text=text, subject=text,
                  status=status, t_start_s=0.0, t_end_s=10.0, speaker_id="spk_0")
        self.store.add_claim(c)
        if entity:
            if self.store.get_entity(entity) is None:
                self.store.upsert_entity(Entity(entity_id=entity, canonical_name=entity))
            self.store.link_mention(cid, entity)
        return c

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()


class TestClaimEvolution(EvolutionTestBase):
    def test_orders_by_publish_date(self):
        self._video("vid:a", "2024-01-01")
        self._video("vid:b", "2026-01-01")
        self._video("vid:c", "2025-01-01")
        self._claim("a.0", "vid:a", "the model is good", entity="ent:model")
        self._claim("b.0", "vid:b", "the model is great", entity="ent:model")
        self._claim("c.0", "vid:c", "the model is fine", entity="ent:model")
        steps = claim_evolution(self.store, entity_id="ent:model")
        self.assertEqual([s.claim.claim_id for s in steps], ["a.0", "c.0", "b.0"])

    def test_undated_sorts_last(self):
        self._video("vid:a", "2024-01-01")
        self._video("vid:b", None)
        self._claim("a.0", "vid:a", "the model is good", entity="ent:model")
        self._claim("b.0", "vid:b", "the model is great", entity="ent:model")
        steps = claim_evolution(self.store, entity_id="ent:model")
        self.assertEqual([s.claim.claim_id for s in steps], ["a.0", "b.0"])

    def test_contradiction_transition_flagged(self):
        self._video("vid:a", "2024-01-01")
        self._video("vid:b", "2026-01-01")
        self._claim("a.0", "vid:a", "scaling will hold", entity="ent:scaling")
        self._claim("b.0", "vid:b", "scaling will not hold", entity="ent:scaling")
        self.store.add_edge("b.0", "CONTRADICTS", "a.0", src_type="Claim", dst_type="Claim")
        steps = claim_evolution(self.store, entity_id="ent:scaling")
        self.assertIsNone(steps[0].relation)
        self.assertEqual(steps[1].relation, "CONTRADICTS")

    def test_correction_transition_flagged(self):
        self._video("vid:a", "2024-01-01")
        self._claim("a.0", "vid:a", "the model has 100 layers", entity="ent:model")
        self._claim("a.1", "vid:a", "actually the model has 96 layers", entity="ent:model")
        self.store.add_edge("a.1", "CORRECTS", "a.0", src_type="Claim", dst_type="Claim")
        steps = claim_evolution(self.store, entity_id="ent:model")
        self.assertEqual(steps[1].relation, "CORRECTS")

    def test_superseded_claim_included_and_flagged(self):
        self._video("vid:a", "2024-01-01")
        self._claim("a.0", "vid:a", "the model has 100 layers", status="superseded",
                    entity="ent:model")
        self._claim("a.1", "vid:a", "the model has 96 layers", entity="ent:model")
        steps = claim_evolution(self.store, entity_id="ent:model")
        ids = [s.claim.claim_id for s in steps]
        self.assertIn("a.0", ids)  # historical record retained
        self.assertTrue(next(s for s in steps if s.claim.claim_id == "a.0").superseded)

    def test_unsupported_claims_excluded(self):
        self._video("vid:a", "2024-01-01")
        self._claim("a.0", "vid:a", "the model is good", status="unsupported",
                    entity="ent:model")
        self._claim("a.1", "vid:a", "the model is great", entity="ent:model")
        steps = claim_evolution(self.store, entity_id="ent:model")
        self.assertEqual([s.claim.claim_id for s in steps], ["a.1"])

    def test_topic_scoped_matches_by_tokens(self):
        self._video("vid:a", "2024-01-01")
        self._video("vid:b", "2026-01-01")
        self._claim("a.0", "vid:a", "scaling laws will hold beyond budgets")
        self._claim("b.0", "vid:b", "the chef cooked pasta for dinner")
        steps = claim_evolution(self.store, topic="scaling laws")
        self.assertEqual([s.claim.claim_id for s in steps], ["a.0"])

    def test_step_carries_deep_link_and_date(self):
        self._video("vid:a", "2024-01-01")
        self._claim("a.0", "vid:a", "the model is good", entity="ent:model")
        steps = claim_evolution(self.store, entity_id="ent:model")
        self.assertEqual(steps[0].published_at, "2024-01-01")
        self.assertTrue(steps[0].deep_link.startswith("https://x/vid:a"))


if __name__ == "__main__":
    unittest.main()
