import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox.backends.embed import HashingEmbedder
from memovox.config import Config
from memovox.loom import Claim, Entity, LoomStore, Moment, Video
from memovox.loom.models import STATUS_SUPERSEDED, STATUS_UNSUPPORTED
from memovox.loom.resolve import link_claim_relations


class LoomTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.config = Config(store=pathlib.Path(self._tmp.name) / "store").ensure()
        self.store = LoomStore(self.config)
        self.emb = HashingEmbedder(dim=256)
        self.video = Video(
            video_id="yt:abc", source_url="https://youtu.be/abc", title="Test talk",
            content_hash="hash1", pipeline_version="0.1.0-phase0",
        )
        self.store.upsert_video(self.video)
        self.m1 = Moment("yt:abc#m0000", "yt:abc", 0.0, 30.0,
                         "neural networks learn via backpropagation and gradients", "spk_0", index=0)
        self.m2 = Moment("yt:abc#m0001", "yt:abc", 30.0, 60.0,
                         "the chef prepared a delicious italian pasta dinner", "spk_0", index=1)
        self.store.add_moment(self.m1, self.emb.embed_one(self.m1.text_for_embedding()))
        self.store.add_moment(self.m2, self.emb.embed_one(self.m2.text_for_embedding()))

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()


class TestVideosMoments(LoomTestBase):
    def test_get_and_list_video(self):
        self.assertEqual(self.store.get_video("yt:abc").title, "Test talk")
        self.assertEqual(len(self.store.list_videos()), 1)

    def test_moments_ordered(self):
        moments = self.store.moments_for_video("yt:abc")
        self.assertEqual([m.index for m in moments], [0, 1])

    def test_visual_embedding_stored_and_counted(self):
        vm = Moment(
            "yt:abc#m0009", "yt:abc", 90.0, 120.0, "a slide is shown", "spk_0",
            visual_caption="bar chart", ocr_text="Loss curve", index=9,
        )
        self.store.add_moment(
            vm, self.emb.embed_one(vm.text_for_embedding()),
            visual_embedding=[0.5, 0.25, 0.75],
        )
        self.assertEqual(self.store.stats()["visual_vectors"], 1)
        self.assertEqual(self.store.get_visual_vector("yt:abc#m0009"), [0.5, 0.25, 0.75])

    def test_visual_embedding_cascades_on_video_delete(self):
        vm = Moment("yt:abc#m0009", "yt:abc", 90.0, 120.0, "slide", "spk_0",
                    ocr_text="Loss curve", index=9)
        self.store.add_moment(vm, self.emb.embed_one("slide Loss curve"),
                              visual_embedding=[0.5, 0.25, 0.75])
        self.store.delete_video("yt:abc")
        self.assertEqual(self.store.stats()["visual_vectors"], 0)

    def test_idempotency_unchanged(self):
        same = Video("yt:abc", "https://youtu.be/abc", "Test talk",
                     content_hash="hash1", pipeline_version="0.1.0-phase0")
        self.assertTrue(self.store.is_unchanged(same))
        changed = Video("yt:abc", None, "Test talk", content_hash="hash2",
                        pipeline_version="0.1.0-phase0")
        self.assertFalse(self.store.is_unchanged(changed))


class TestSearch(LoomTestBase):
    def test_lexical_search(self):
        hits = self.store.lexical_search("backpropagation")
        self.assertTrue(hits)
        self.assertEqual(hits[0][0], "yt:abc#m0000")

    def test_vector_search_ranks_relevant_first(self):
        q = self.emb.embed_one("deep learning neural network gradients")
        hits = self.store.vector_search(q, top_k=2)
        self.assertEqual(hits[0][0], "yt:abc#m0000")

    def test_vector_search_dim_mismatch_skipped(self):
        # A 4-dim query cannot match 256-dim stored vectors -> no results, no crash.
        self.assertEqual(self.store.vector_search([0.1, 0.2, 0.3, 0.4]), [])

    def test_stored_vectors_are_unit_normalized(self):
        from memovox.vectormath import norm

        m = Moment("yt:abc#m0002", "yt:abc", 60.0, 90.0, "x", "spk_0", index=2)
        self.store.add_moment(m, [3.0, 4.0])  # non-unit input; stored must be unit
        stored = dict(self.store.moment_vectors())["yt:abc#m0002"]
        self.assertAlmostEqual(norm(stored), 1.0, places=5)
        self.assertAlmostEqual(stored[0], 0.6, places=5)
        self.assertAlmostEqual(stored[1], 0.8, places=5)

    def test_zero_vector_not_normalized_and_search_guarded(self):
        m = Moment("yt:abc#m0003", "yt:abc", 90.0, 120.0, "z", "spk_0", index=3)
        self.store.add_moment(m, [0.0, 0.0])  # zero vector survives (no div-by-zero)
        stored = dict(self.store.moment_vectors())["yt:abc#m0003"]
        self.assertEqual(stored, [0.0, 0.0])
        self.assertEqual(self.store.vector_search([0.0, 0.0]), [])  # zero-query guard

    def test_text_and_visual_vectors_are_space_tagged(self):
        from memovox.errors import VectorSpaceError

        vm = Moment("yt:abc#m0005", "yt:abc", 100.0, 130.0, "a slide", "spk_0", index=5)
        self.store.add_moment(vm, self.emb.embed_one("a slide"),
                              visual_embedding=[0.5] * 256)
        q = self.emb.embed_one("neural networks backpropagation")
        ids = [m for m, _ in self.store.vector_search(q, 5, space="text")]
        self.assertIn("yt:abc#m0000", ids)            # text rows scored
        # the text index serves only the 'text' space: a cross-space request raises
        with self.assertRaises(VectorSpaceError):
            self.store.vector_search(q, 5, space="visual_sig")

    def test_visual_vector_stored_raw_not_normalized(self):
        # space tagging must NOT normalize visual signatures (they live in their
        # own space and are scored by cosine, which is scale-invariant).
        vm = Moment("yt:abc#m0006", "yt:abc", 130.0, 160.0, "slide", "spk_0", index=6)
        self.store.add_moment(vm, self.emb.embed_one("slide"),
                              visual_embedding=[0.5, 0.25, 0.75])
        self.assertEqual(self.store.get_visual_vector("yt:abc#m0006"), [0.5, 0.25, 0.75])

    def test_vector_search_matches_legacy_cosine_topk(self):
        # The new normalized+dot path must rank identically to legacy per-row
        # cosine over the ORIGINAL (raw) embeddings.
        from memovox.vectormath import cosine

        raw = {
            "yt:abc#m0000": self.emb.embed_one(self.m1.text_for_embedding()),
            "yt:abc#m0001": self.emb.embed_one(self.m2.text_for_embedding()),
        }
        q = self.emb.embed_one("deep learning neural network gradients")
        ref = sorted(raw, key=lambda mid: cosine(q, raw[mid]), reverse=True)
        new_ids = [mid for mid, _ in self.store.vector_search(q, top_k=2)]
        self.assertEqual(new_ids, ref)


class TestClaimsGraph(LoomTestBase):
    def test_claim_roundtrip(self):
        claim = Claim(
            claim_id="yt:abc#m0000.c00", moment_id="yt:abc#m0000", video_id="yt:abc",
            text="neural networks learn via backpropagation", subject="neural networks",
            predicate="learn via", object="backpropagation", claim_type="FACT",
            entailment_score=0.9, t_start_s=0.0, t_end_s=30.0, speaker_id="spk_0",
            qualifiers={"hedge": False},
        )
        self.store.add_claim(claim)
        got = self.store.get_claim("yt:abc#m0000.c00")
        self.assertEqual(got.object, "backpropagation")
        self.assertEqual(got.qualifiers, {"hedge": False})
        self.assertEqual(len(self.store.claims_for_video("yt:abc")), 1)

    def test_edges_neighbors(self):
        self.store.add_edge("spk_0", "STATES", "yt:abc#m0000.c00",
                            src_type="Speaker", dst_type="Claim", video_id="yt:abc")
        out = self.store.neighbors("spk_0", rel="STATES")
        self.assertEqual(out[0]["dst"], "yt:abc#m0000.c00")
        # idempotent: inserting the same edge again does not duplicate
        self.store.add_edge("spk_0", "STATES", "yt:abc#m0000.c00", video_id="yt:abc")
        self.assertEqual(len(self.store.neighbors("spk_0", rel="STATES")), 1)


class TestClaimSupersede(LoomTestBase):
    def test_supersede_versions_old_claim(self):
        c_old = Claim(
            claim_id="yt:abc#m0000.c00", moment_id="yt:abc#m0000", video_id="yt:abc",
            text="the model has 100M parameters", t_start_s=0.0, t_end_s=30.0,
        )
        c_new = Claim(
            claim_id="yt:abc#m0001.c00", moment_id="yt:abc#m0001", video_id="yt:abc",
            text="the model has 200M parameters", t_start_s=30.0, t_end_s=60.0,
        )
        self.store.add_claim(c_old)
        self.store.add_claim(c_new)

        self.store.supersede_claim("yt:abc#m0000.c00", "yt:abc#m0001.c00")

        # Old claim is versioned: status flipped + superseded_by points to new.
        got_old = self.store.get_claim("yt:abc#m0000.c00")
        self.assertIsNotNone(got_old)  # never deleted — still fetchable
        self.assertEqual(got_old.status, STATUS_SUPERSEDED)
        self.assertEqual(got_old.status, "superseded")
        self.assertEqual(got_old.superseded_by, "yt:abc#m0001.c00")

        # Superseded claim drops out of default (committed) queries.
        vid_ids = [c.claim_id for c in self.store.claims_for_video("yt:abc")]
        self.assertNotIn("yt:abc#m0000.c00", vid_ids)
        self.assertIn("yt:abc#m0001.c00", vid_ids)

        list_ids = [c.claim_id for c in self.store.list_claims()]
        self.assertNotIn("yt:abc#m0000.c00", list_ids)
        self.assertIn("yt:abc#m0001.c00", list_ids)

        # New claim is untouched.
        got_new = self.store.get_claim("yt:abc#m0001.c00")
        self.assertEqual(got_new.status, "committed")
        self.assertIsNone(got_new.superseded_by)


class TestEntities(LoomTestBase):
    def test_entity_roundtrip(self):
        ent = Entity(entity_id="ent:transformer", canonical_name="Transformer",
                     type="concept", wikidata_qid="Q1", aliases=["Transformer"])
        self.store.upsert_entity(ent)
        got = self.store.get_entity("ent:transformer")
        self.assertEqual(got.canonical_name, "Transformer")
        self.assertEqual(got.type, "concept")
        self.assertEqual(got.wikidata_qid, "Q1")
        self.assertEqual(got.aliases, ["Transformer"])
        self.assertIsNone(self.store.get_entity("ent:missing"))

    def test_list_entities(self):
        self.store.upsert_entity(Entity("ent:b", "Beta"))
        self.store.upsert_entity(Entity("ent:a", "Alpha"))
        ids = [e.entity_id for e in self.store.list_entities()]
        self.assertEqual(ids, ["ent:a", "ent:b"])  # ORDER BY entity_id

    def test_upsert_entity_merges_aliases_and_preserves_mentions(self):
        # Re-upserting a shared entity must NOT cascade-delete existing mentions
        # (the W2.3 cross-video unification bug) and SHOULD accumulate aliases.
        self.store.add_claim(Claim("yt:abc#m0000.c00", "yt:abc#m0000", "yt:abc", "x"))
        self.store.upsert_entity(Entity("ent:x", "X", aliases=["X"]))
        self.store.link_mention("yt:abc#m0000.c00", "ent:x")
        self.store.upsert_entity(Entity("ent:x", "X", aliases=["Xs"]))
        self.assertEqual(self.store.entity_mentions("ent:x"), ["yt:abc#m0000.c00"])
        self.assertEqual(self.store.get_entity("ent:x").aliases, ["X", "Xs"])


class TestGetClaims(LoomTestBase):
    def test_get_claims_batch_roundtrip(self):
        c0 = Claim("yt:abc#m0000.c00", "yt:abc#m0000", "yt:abc", "first claim")
        c1 = Claim("yt:abc#m0001.c00", "yt:abc#m0001", "yt:abc", "second claim")
        self.store.add_claim(c0)
        self.store.add_claim(c1)
        got = self.store.get_claims(["yt:abc#m0001.c00", "yt:abc#m0000.c00"])
        # Order follows the requested id order; missing ids are skipped.
        self.assertEqual([c.claim_id for c in got],
                         ["yt:abc#m0001.c00", "yt:abc#m0000.c00"])
        self.assertEqual(self.store.get_claims([]), [])
        self.assertEqual(self.store.get_claims(["ent:nope"]), [])


class TestClaimRelations(LoomTestBase):
    """W3.1 — ELABORATES (intra-Moment, same-speaker) + CORRECTS (nearest sharer)."""

    def _claim(self, cid, moment_id, text, *, subject="", claim_type="FACT",
               status="committed", speaker="spk_0", t0=0.0, t1=1.0):
        c = Claim(
            claim_id=cid, moment_id=moment_id, video_id="yt:abc", text=text,
            subject=subject, claim_type=claim_type, status=status,
            t_start_s=t0, t_end_s=t1, speaker_id=speaker,
        )
        self.store.add_claim(c)
        return c

    def test_elaborates_within_moment_same_speaker(self):
        a = self._claim("yt:abc#m0000.c00", "yt:abc#m0000", "GPT is a model.", t0=0.0, t1=2.0)
        b = self._claim("yt:abc#m0000.c01", "yt:abc#m0000", "It uses attention.", t0=2.0, t1=4.0)
        link_claim_relations(self.store, [a, b])
        out = self.store.neighbors(a.claim_id, rel="ELABORATES")
        self.assertEqual([e["dst"] for e in out], [b.claim_id])
        # Provenance spans a.t_start_s .. b.t_end_s, carries the video id.
        edge = out[0]
        self.assertEqual(edge["video_id"], "yt:abc")
        self.assertEqual(edge["t_start_s"], 0.0)
        self.assertEqual(edge["t_end_s"], 4.0)
        self.assertEqual(edge["src_type"], "Claim")
        self.assertEqual(edge["dst_type"], "Claim")

    def test_no_elaborates_across_moments(self):
        a = self._claim("yt:abc#m0000.c00", "yt:abc#m0000", "GPT is a model.")
        b = self._claim("yt:abc#m0001.c00", "yt:abc#m0001", "Pasta is tasty.")
        link_claim_relations(self.store, [a, b])
        self.assertEqual(self.store.neighbors(a.claim_id, rel="ELABORATES"), [])

    def test_no_elaborates_across_speaker_boundary(self):
        a = self._claim("yt:abc#m0000.c00", "yt:abc#m0000", "GPT is a model.", speaker="spk_0")
        b = self._claim("yt:abc#m0000.c01", "yt:abc#m0000", "It uses attention.", speaker="spk_1")
        link_claim_relations(self.store, [a, b])
        self.assertEqual(self.store.neighbors(a.claim_id, rel="ELABORATES"), [])

    def test_elaborates_skips_uncommitted_claims(self):
        a = self._claim("yt:abc#m0000.c00", "yt:abc#m0000", "GPT is a model.")
        b = self._claim("yt:abc#m0000.c01", "yt:abc#m0000", "It uses attention.",
                        status=STATUS_UNSUPPORTED)
        link_claim_relations(self.store, [a, b])
        self.assertEqual(self.store.neighbors(a.claim_id, rel="ELABORATES"), [])

    def test_corrects_links_to_nearest_prior_sharer(self):
        # Two earlier claims share entity "GPT"; the CORRECTION must link the
        # NEAREST (latest) prior sharer, not the earliest. m0002 is an extra
        # Moment FK parent beyond the base fixture's m0000/m0001.
        self.store.add_moment(
            Moment("yt:abc#m0002", "yt:abc", 60.0, 90.0, "correction span", "spk_0", index=2),
            self.emb.embed_one("correction span"),
        )
        far = self._claim("yt:abc#m0000.c00", "yt:abc#m0000",
                          "GPT has 100 layers.", subject="GPT", t0=0.0, t1=2.0)
        near = self._claim("yt:abc#m0001.c00", "yt:abc#m0001",
                           "GPT was trained in 2020.", subject="GPT", t0=10.0, t1=12.0)
        corr = self._claim("yt:abc#m0002.c00", "yt:abc#m0002",
                           "Actually, GPT has 96 layers.", subject="GPT",
                           claim_type="CORRECTION", t0=20.0, t1=22.0)
        link_claim_relations(self.store, [far, near, corr])
        out = self.store.neighbors(corr.claim_id, rel="CORRECTS")
        self.assertEqual([e["dst"] for e in out], [near.claim_id])
        self.assertEqual(out[0]["video_id"], "yt:abc")
        self.assertEqual(out[0]["t_start_s"], 20.0)
        self.assertEqual(out[0]["t_end_s"], 22.0)

    def test_corrects_via_subject_token_overlap(self):
        prior = self._claim("yt:abc#m0000.c00", "yt:abc#m0000",
                            "the learning rate is small.", subject="learning rate",
                            t0=0.0, t1=2.0)
        corr = self._claim("yt:abc#m0001.c00", "yt:abc#m0001",
                           "i misspoke, the learning rate is large.",
                           subject="learning rate", claim_type="CORRECTION",
                           t0=10.0, t1=12.0)
        link_claim_relations(self.store, [prior, corr])
        out = self.store.neighbors(corr.claim_id, rel="CORRECTS")
        self.assertEqual([e["dst"] for e in out], [prior.claim_id])

    def test_corrects_ignores_trivial_stopword_subject_overlap(self):
        # Precision boundary: "the model" (CORRECTION) and "the method" (prior)
        # share ONLY the stopword "the". A content-word check must NOT link them
        # (raw-tokenize overlap would emit a spurious CORRECTS). Neither subject
        # surfaces an entity, so the entity leg also stays silent.
        prior = self._claim("yt:abc#m0000.c00", "yt:abc#m0000",
                            "the method is slow.", subject="the method",
                            t0=0.0, t1=2.0)
        corr = self._claim("yt:abc#m0001.c00", "yt:abc#m0001",
                           "actually, the model is fast.", subject="the model",
                           claim_type="CORRECTION", t0=10.0, t1=12.0)
        link_claim_relations(self.store, [prior, corr])
        self.assertEqual(self.store.neighbors(corr.claim_id, rel="CORRECTS"), [])

    def test_corrects_emits_nothing_when_no_sharer(self):
        prior = self._claim("yt:abc#m0000.c00", "yt:abc#m0000",
                            "Pasta is tasty.", subject="pasta", t0=0.0, t1=2.0)
        corr = self._claim("yt:abc#m0001.c00", "yt:abc#m0001",
                           "Actually, BERT is bidirectional.", subject="BERT",
                           claim_type="CORRECTION", t0=10.0, t1=12.0)
        link_claim_relations(self.store, [prior, corr])
        self.assertEqual(self.store.neighbors(corr.claim_id, rel="CORRECTS"), [])

    def test_relations_idempotent(self):
        a = self._claim("yt:abc#m0000.c00", "yt:abc#m0000", "GPT is a model.", t0=0.0, t1=2.0)
        b = self._claim("yt:abc#m0000.c01", "yt:abc#m0000", "It uses attention.", t0=2.0, t1=4.0)
        link_claim_relations(self.store, [a, b])
        link_claim_relations(self.store, [a, b])
        self.assertEqual(len(self.store.neighbors(a.claim_id, rel="ELABORATES")), 1)


class TestDeleteCascade(LoomTestBase):
    def test_delete_video_cascades(self):
        self.store.add_claim(Claim("yt:abc#m0000.c00", "yt:abc#m0000", "yt:abc", "x"))
        self.assertTrue(self.store.delete_video("yt:abc"))
        self.assertIsNone(self.store.get_moment("yt:abc#m0000"))
        self.assertEqual(self.store.moments_for_video("yt:abc"), [])
        self.assertEqual(self.store.lexical_search("backpropagation"), [])
        self.assertEqual(self.store.vector_search(self.emb.embed_one("neural")), [])
        stats = self.store.stats()
        self.assertEqual(stats["moments"], 0)
        self.assertEqual(stats["claims"], 0)


if __name__ == "__main__":
    unittest.main()
