import pytest
from xunit_rag.config import Config
from xunit_rag.schemas import Leaf, Node
from xunit_rag.utils import digest


class TestTokenizer:
    __test__ = False
    policy = {"test_only": True}

    def count_tokens(self, text):
        return len(text.split())


@pytest.fixture
def config(tmp_path):
    config = Config(root=tmp_path)
    (tmp_path / "docs").mkdir()
    config.initialize()
    return config


@pytest.fixture
def tokenizer():
    return TestTokenizer()


def make_leaf(index, text=None):
    text = text or f"Independent source passage {index} about Test Stub and Mock Object."
    return Leaf(
        leaf_id=f"l_{index}",
        doc_id="d_1",
        source_path="source.txt",
        original_name="source.txt",
        source_sha256="source_hash",
        text=text,
        embedding_text=text,
        content_sha256=digest(text),
        language="en",
        headings=[],
        provenance=[{"docling_ref": f"#/texts/{index}"}],
        pdf_pages=[],
        location={"chunk_order": index},
        order=index,
        token_policy={"test_only": True},
    )


def make_node(index, text=None):
    leaf = make_leaf(index, text)
    return Node(
        node_id=leaf.leaf_id,
        node_type="leaf",
        level=0,
        text=leaf.text,
        embedding_text=leaf.text,
        content_sha256=leaf.content_sha256,
        source_leaf_ids=[leaf.leaf_id],
        doc_ids=[leaf.doc_id],
        source_paths=[leaf.source_path],
        provenance=[
            {"leaf_id": leaf.leaf_id, "source_path": leaf.source_path, "pdf_pages": [], "location": {}}
        ],
        embedding_fingerprint="test_fingerprint",
    )
