"""W1 — topic induction + ABOUT edges (Phase 3, spec §4.7 / §6).

Greedy cosine clustering of moments into emergent Topics, with a provenanced
``(Moment)-[:ABOUT]->(Topic)`` edge per member. Free path: clusters over the
PERSISTED text vectors (hashing embedder), so it is deterministic and needs no
model. These tests exercise the load-bearing properties: similar moments group
(across videos), dissimilar moments split, ids are deterministic + idempotent,
and the topic_id / ABOUT edge / moment_count are persisted.
"""

import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox.backends.embed import HashingEmbedder
from memovox.config import Config, Settings
from memovox.loom import LoomStore, Moment, Video
from memovox.loom.topics import induce_topics
from memovox.util import slugify, tokenize


class TopicTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.config = Config(store=pathlib.Path(self._tmp.name) / "store").ensure()
        self.store = LoomStore(self.config)
        self.emb = HashingEmbedder(dim=256)
        # Two videos. talk_a and talk_b each carry a near-identical "scaling
        # laws / transformer" moment (these must land in ONE topic across the two
        # videos) plus a clearly off-topic "cooking" moment.
        for vid in ("vid:a", "vid:b"):
            self.store.upsert_video(Video(video_id=vid, source_url=f"https://x/{vid}",
                                          title=vid, content_hash=vid))
        self._add("vid:a", 0, "Scaling laws govern how transformer language models improve.")
        self._add("vid:a", 1, "The chef cooked a delicious italian pasta dinner tonight.")
        self._add("vid:b", 0, "Scaling laws govern how transformer language models improve over time.")

    def _add(self, vid, idx, text):
        m = Moment(f"{vid}#m{idx:04d}", vid, float(idx * 30), float(idx * 30 + 30),
                   text, "spk_0", index=idx)
        self.store.add_moment(m, self.emb.embed_one(text))
        return m

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()


class TestInduceTopics(TopicTestBase):
    def test_similar_moments_share_a_topic_across_videos(self):
        induce_topics(self.store)
        ta = self.store.get_moment("vid:a#m0000").topic_id
        tb = self.store.get_moment("vid:b#m0000").topic_id
        self.assertIsNotNone(ta)
        self.assertEqual(ta, tb)  # cross-video same-topic merge

    def test_dissimilar_moment_gets_its_own_topic(self):
        induce_topics(self.store)
        scaling = self.store.get_moment("vid:a#m0000").topic_id
        cooking = self.store.get_moment("vid:a#m0001").topic_id
        self.assertIsNotNone(cooking)
        self.assertNotEqual(scaling, cooking)

    def test_topic_id_and_label_are_content_word_based(self):
        topics = induce_topics(self.store)
        scaling = self.store.get_moment("vid:a#m0000").topic_id
        # Deterministic slug id derived from the cluster's top content tokens.
        self.assertTrue(scaling.startswith("topic:"))
        topic = next(t for t in topics if t.topic_id == scaling)
        # The label is built only from content words present in the clustered
        # moments, and the id is the slug of that label (deterministic).
        member_text = " ".join(m.text_for_embedding()
                               for m in self.store.moments_for_topic(scaling))
        member_tokens = set(tokenize(member_text))
        label_tokens = tokenize(topic.label)
        self.assertTrue(label_tokens)
        self.assertTrue(all(t in member_tokens for t in label_tokens))
        self.assertEqual(scaling, "topic:" + slugify(topic.label))

    def test_about_edges_emitted_with_provenance(self):
        induce_topics(self.store)
        scaling = self.store.get_moment("vid:a#m0000").topic_id
        edges = self.store.neighbors("vid:a#m0000", rel="ABOUT")
        self.assertEqual([e["dst"] for e in edges], [scaling])
        e = edges[0]
        self.assertEqual(e["src_type"], "Moment")
        self.assertEqual(e["dst_type"], "Topic")
        self.assertEqual(e["video_id"], "vid:a")
        self.assertEqual(e["t_start_s"], 0.0)
        self.assertEqual(e["t_end_s"], 30.0)

    def test_moment_count_and_listing(self):
        induce_topics(self.store)
        scaling = self.store.get_moment("vid:a#m0000").topic_id
        topic = next(t for t in self.store.list_topics() if t.topic_id == scaling)
        self.assertEqual(topic.moment_count, 2)  # vid:a#m0000 + vid:b#m0000
        members = self.store.moments_for_topic(scaling)
        self.assertEqual({m.moment_id for m in members},
                         {"vid:a#m0000", "vid:b#m0000"})

    def test_idempotent_rerun(self):
        induce_topics(self.store)
        topics1 = {t.topic_id for t in self.store.list_topics()}
        edges1 = len(self.store.edges(rel="ABOUT"))
        induce_topics(self.store)
        topics2 = {t.topic_id for t in self.store.list_topics()}
        edges2 = len(self.store.edges(rel="ABOUT"))
        self.assertEqual(topics1, topics2)
        self.assertEqual(edges1, edges2)  # no duplicate ABOUT edges

    def test_min_size_drops_small_topics(self):
        # With topic_min_size=2 the lone cooking moment forms no topic.
        induce_topics(self.store, settings=Settings(topic_min_size=2))
        self.assertIsNone(self.store.get_moment("vid:a#m0001").topic_id)
        # The 2-member scaling topic still lands.
        self.assertIsNotNone(self.store.get_moment("vid:a#m0000").topic_id)


if __name__ == "__main__":
    unittest.main()
