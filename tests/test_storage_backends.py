"""M0.2 W1 — storage-backend ABCs + SQLite default, parity-locked no-op refactor.

The four-index store gains a backend seam (VectorIndex / LexicalIndex / GraphStore)
mirroring the model-backend registry. This commit is purely structural: LoomStore
delegates to SQLite-backed defaults, and the parity test proves results are
byte-identical whether reached via LoomStore or a freshly-constructed index on the
same connection.
"""

from __future__ import annotations

import inspect
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memovox.backends.base import Backend
from memovox.backends.embed import HashingEmbedder
from memovox.config import Config
from memovox.errors import BackendUnavailable
from memovox.loom import LoomStore, Moment, Video
from memovox.loom.backends import (
    GraphStore,
    LexicalIndex,
    VectorIndex,
    get_graph_store,
    get_lexical_index,
    get_vector_index,
)
from memovox.loom.backends.sqlite import (
    SqliteGraphStore,
    SqliteLexicalIndex,
    SqliteVectorIndex,
)


class AbcShapeTest(unittest.TestCase):
    def test_abcs_have_is_available_classmethod(self):
        for abc in (VectorIndex, LexicalIndex, GraphStore):
            self.assertTrue(hasattr(abc, "is_available"))
            self.assertTrue(inspect.ismethod(abc.is_available))  # classmethod bound
            self.assertTrue(abc.is_available())  # stdlib SQLite always available-ish

    def test_registry_auto_and_sqlite_resolve_to_sqlite(self):
        self.assertIsInstance(get_vector_index("auto", conn=None), SqliteVectorIndex)
        self.assertIsInstance(get_vector_index("sqlite", conn=None), SqliteVectorIndex)
        self.assertIsInstance(get_lexical_index("auto", conn=None, fts=False), SqliteLexicalIndex)
        self.assertIsInstance(get_graph_store("auto", conn=None), SqliteGraphStore)

    def test_unknown_backend_raises(self):
        with self.assertRaises(BackendUnavailable):
            get_vector_index("lance", conn=None)


class ParityTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.config = Config(store=pathlib.Path(self._tmp.name) / "store").ensure()
        self.store = LoomStore(self.config)
        self.emb = HashingEmbedder(dim=256)
        self.store.upsert_video(Video(video_id="v", source_url="https://x/v",
                                      title="v", content_hash="v"))
        for i in range(12):
            text = f"machine learning models training run number {i}"
            mid = f"v#m{i:04d}"
            self.store.add_moment(Moment(mid, "v", float(i), float(i) + 5, text, "spk_0",
                                         index=i), self.emb.embed_one(text))
        self.store.add_edge("v#m0000", "PRECEDES", "v#m0001",
                            src_type="Moment", dst_type="Moment", video_id="v")

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def test_vector_search_parity(self):
        q = self.emb.embed_one("models training")
        via_store = self.store.vector_search(q, 5)
        via_index = SqliteVectorIndex(self.store.conn).search(q, 5)
        self.assertEqual(via_store, via_index)

    def test_lexical_search_parity(self):
        via_store = self.store.lexical_search("models training", 5)
        via_index = SqliteLexicalIndex(self.store.conn, self.store.fts).search("models training", 5)
        self.assertEqual(via_store, via_index)

    def test_graph_parity(self):
        via_store = self.store.neighbors("v#m0000", rel="PRECEDES")
        via_index = SqliteGraphStore(self.store.conn).neighbors("v#m0000", rel="PRECEDES")
        self.assertEqual(via_store, via_index)
        self.assertEqual(self.store.edges(rel="PRECEDES"),
                         SqliteGraphStore(self.store.conn).edges(rel="PRECEDES"))


if __name__ == "__main__":
    unittest.main()
