"""Round 3 (Option A follow-up): semantic candidate generation.

The free/lexical path generates contradiction/consensus candidate pairs by token overlap
(bucket-blocking + a >=3-shared / Jaccard>=0.5 precision gate). That STRUCTURALLY filters
out the low-lexical-overlap pairs where a real [embed]/[nli] backend would add value —
proven on the iterE upgraded run (the planted sat-fat pair has Jaccard 0.087, dropped
before DeBERTa sees it). With a SEMANTIC embedder + SEMANTIC NLI, candidates must instead
come from the induced topic clusters (cross-video claims sharing a topic), judged by the
precise NLI with NO lexical Jaccard prefilter. The free path must stay byte-identical.
"""
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox.backends.base import NLIBackend, NLIResult
from memovox.config import Config
from memovox.loom import LoomStore, Moment, Video
from memovox.loom.consolidate import find_contradictions
from memovox.loom.models import Claim


class _FakePreciseNLI(NLIBackend):
    """Stands in for DeBERTa: precise (is_semantic=True), flags a known pair regardless of
    lexical overlap. The point under test is candidate GENERATION + gating, not the NLI."""

    is_semantic = True
    name = "fake-precise"

    def __init__(self, contradictions):
        self._contras = {frozenset(p) for p in contradictions}

    @classmethod
    def is_available(cls) -> bool:
        return True

    def classify(self, premise: str, hypothesis: str) -> NLIResult:
        if frozenset((premise, hypothesis)) in self._contras:
            return NLIResult("contradiction", 0.0, 0.05, 0.95)
        return NLIResult("neutral", 0.0, 1.0, 0.0)


class TestSemanticContradictionCandidates(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = LoomStore(Config(store=pathlib.Path(self._tmp.name) / "s").ensure())
        for v in ("yt:a", "yt:b"):
            self.store.upsert_video(Video(v, f"https://youtu.be/{v[3:]}", "talk"))
        # Two contradictory claims that share NO content tokens (Jaccard ~0) but ARE about
        # the same subject -> the lexical path can never pair them; a semantic path must.
        self.a_text = "the new policy will clearly boost national output next year"
        self.b_text = "such measures only deepen the downturn and unemployment"
        self._add("yt:a#m0.c0", "yt:a#m0", "yt:a", self.a_text)
        self._add("yt:b#m0.c0", "yt:b#m0", "yt:b", self.b_text)
        # induce_topics would stamp these (semantically similar) moments into one cluster;
        # simulate that result directly.
        self.store.set_moment_topic("yt:a#m0", "topic:economy")
        self.store.set_moment_topic("yt:b#m0", "topic:economy")
        self.nli = _FakePreciseNLI([(self.a_text, self.b_text)])

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def _add(self, cid, mid, vid, text):
        self.store.add_moment(Moment(mid, vid, 0.0, 1.0, text, index=0))
        self.store.add_claim(Claim(cid, mid, vid, text, t_start_s=0.0, t_end_s=1.0))

    def test_dense_mode_surfaces_low_jaccard_same_topic_contradiction(self):
        pairs = find_contradictions(self.store, nli=self.nli, dense=True, write_edges=False)
        flagged = {frozenset((p.claim_a.claim_id, p.claim_b.claim_id)) for p in pairs}
        self.assertIn(frozenset(("yt:a#m0.c0", "yt:b#m0.c0")), flagged,
                      "semantic path missed a same-topic, low-Jaccard cross-video contradiction")

    def test_lexical_mode_drops_low_jaccard_pair(self):
        # The existing free/lexical path (token-overlap candidates + Jaccard gate) must NOT
        # surface it — there is no shared token to even generate the candidate.
        pairs = find_contradictions(self.store, nli=self.nli, dense=False, write_edges=False)
        flagged = {frozenset((p.claim_a.claim_id, p.claim_b.claim_id)) for p in pairs}
        self.assertNotIn(frozenset(("yt:a#m0.c0", "yt:b#m0.c0")), flagged)


class TestSemanticConsensusCandidates(unittest.TestCase):
    """Consensus dense path: synonym claims that share NO content token but ARE the same
    topic + near-identical embeddings must cluster (the synthesis_synonyms limit). The
    lexical path can't pair them (no shared token); the dense path blocks on topic + cosine."""

    def _claims(self):
        return [
            Claim("yt:a#m0.c0", "yt:a#m0", "yt:a", "AGI is coming very soon", subject="agi"),
            Claim("yt:b#m0.c0", "yt:b#m0", "yt:b",
                  "artificial general intelligence arrives imminently", subject="agi"),
        ]

    def test_dense_groups_no_token_overlap_synonyms_via_cosine(self):
        from memovox.loom.consensus import partition_claims
        c = self._claims()
        vectors = {c[0].claim_id: [1.0, 0.0, 0.05], c[1].claim_id: [0.98, 0.0, 0.10]}
        topic = {c[0].claim_id: "topic:agi", c[1].claim_id: "topic:agi"}
        groups, xv = partition_claims(c, cosine=0.9, vectors=vectors,
                                      dense=True, claim_topic=topic)
        self.assertEqual(len(groups), 1, "dense path did not merge no-overlap synonyms")
        self.assertEqual(len(xv), 1)

    def test_lexical_path_cannot_group_no_token_overlap(self):
        from memovox.loom.consensus import partition_claims
        c = self._claims()
        vectors = {c[0].claim_id: [1.0, 0.0, 0.05], c[1].claim_id: [0.98, 0.0, 0.10]}
        groups, xv = partition_claims(c, cosine=0.9, vectors=vectors)  # lexical (no dense)
        self.assertEqual(len(groups), 2)  # never paired -> two singletons
        self.assertEqual(xv, [])


class TestBackendSemanticFlags(unittest.TestCase):
    """The dense path is gated on BOTH the embedder and the NLI being semantic, so each
    backend must declare is_semantic — hashing/lexical False, BGE-M3/DeBERTa True."""

    def test_embedder_flags(self):
        from memovox.backends.embed import HashingEmbedder, SentenceTransformerEmbedder
        self.assertFalse(HashingEmbedder.is_semantic)
        self.assertTrue(SentenceTransformerEmbedder.is_semantic)

    def test_nli_flags(self):
        from memovox.backends.nli import LexicalNLI, TransformersNLI
        self.assertFalse(LexicalNLI.is_semantic)
        self.assertTrue(TransformersNLI.is_semantic)


if __name__ == "__main__":
    unittest.main()
