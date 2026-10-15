import numpy as np
import pytest
from xunit_rag.raptor import RaptorBuilder
from xunit_rag.schemas import SummaryAnswer
from xunit_rag.storage import IndexStore, dependency_versions, storage_path_policy, validate_graph
from xunit_rag.tokenization import PromptBudget
from xunit_rag.utils import digest, write_jsonl
from conftest import make_leaf, make_node


class FakeEmbeddings:
    fingerprint = "test_fingerprint"

    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


class FakeSession:
    def __init__(self, config):
        self.model_info = {"models": {config.llm_model: {"digest": "test_llm_digest"}}}
        self.calls = 0

    def generate(self, *args):
        self.calls += 1
        return SummaryAnswer(text="Faithful generated summary of the children.")


def builder(config, tokenizer, directory):
    return RaptorBuilder(
        config,
        FakeEmbeddings(),
        FakeSession(config),
        PromptBudget(config, tokenizer),
        "Summarize only children.",
        directory,
        progress=lambda _: None,
    )


def test_small_terminal_and_cache_resume(config, tokenizer):
    leaves = [make_leaf(1), make_leaf(2)]
    first = builder(config, tokenizer, config.data / "builds" / "test")
    nodes, _, _ = first.build(leaves)
    assert validate_graph(nodes, [leaf.leaf_id for leaf in leaves])["summaries"] == 1
    assert first.session.calls == 1
    second = builder(config, tokenizer, config.data / "builds" / "test")
    again, _, _ = second.build(leaves)
    assert second.session.calls == 0 and [n.node_id for n in nodes] == [n.node_id for n in again]
    single = builder(config, tokenizer, config.data / "builds" / "single")
    one, _, _ = single.build([make_leaf(1)])
    assert len(one) == 1 and single.session.calls == 0


def test_soft_memberships_multiple_parents_and_propagation(config, tokenizer):
    tree = builder(config, tokenizer, config.data / "builds" / "soft")

    def groups(nodes, vectors, forced_components=None):
        return [[nodes[0], nodes[1]], [nodes[1], nodes[2]]] if len(nodes) == 3 else [nodes]

    tree.gmm_groups = groups
    nodes, _, _ = tree.build([make_leaf(1), make_leaf(2), make_leaf(3)])
    assert validate_graph(nodes)["summary_levels"] == 2
    assert len(next(n for n in nodes if n.node_id == "l_2").parent_node_ids) == 2
    assert set(max(nodes, key=lambda n: n.level).source_leaf_ids) == {"l_1", "l_2", "l_3"}


def test_cycles_and_missing_children_rejected(config, tokenizer):
    tree = builder(config, tokenizer, config.data / "builds" / "bad")
    nodes, _, _ = tree.build([make_leaf(1), make_leaf(2)])
    parent = next(n for n in nodes if n.node_type == "summary")
    parent.child_node_ids.append(parent.node_id)
    with pytest.raises(ValueError):
        validate_graph(nodes)


def test_no_progress_does_not_claim_complete(config, tokenizer):
    tree = builder(config, tokenizer, config.data / "builds" / "no_progress")
    tree.gmm_groups = lambda nodes, vectors: [[node] for node in nodes]
    with pytest.raises(ValueError, match="sem compressão"):
        tree.build([make_leaf(1), make_leaf(2)])
    assert not (tree.build_dir / "COMPLETE").exists()


def test_real_gmm_memberships_do_not_lose_nodes(config, tokenizer):
    tree = builder(config, tokenizer, config.data / "builds" / "numeric")
    nodes = [make_node(i) for i in range(4)]
    groups = tree.gmm_groups(nodes, np.asarray([[1, 0], [0.99, 0.01], [0, 1], [0.01, 0.99]]))
    assert {n.node_id for group in groups for n in group} == {n.node_id for n in nodes}


def test_chroma_summary_retrieval_and_atomic_promotion(config, tokenizer, monkeypatch):
    tree = builder(config, tokenizer, config.data / "builds" / "test_chroma")
    leaves = [make_leaf(1), make_leaf(2)]
    nodes, vectors, reason = tree.build(leaves)
    write_jsonl(config.data / "processed" / "ingest_corpus.jsonl", leaves)
    ingestion = {
        "corpus_hash": digest([leaf.model_dump() for leaf in leaves]),
        "inventory_hash": digest([]),
        "files": {"source.txt": {"status": "ingested"}},
    }
    build_contract = {
        "storage_path_policy": storage_path_policy(),
        "profile": config.profile(),
        "dependencies": dependency_versions(),
        "embedding_policy": {"dimension": 3},
        "embedding_fingerprint": "test_fingerprint",
    }
    store = IndexStore(config)
    manifest = store.persist_and_promote("test_chroma", nodes, vectors, ingestion, build_contract, reason)
    assert store.validate()["status"] == "valid"
    summary_id = next(n.node_id for n in nodes if n.node_type == "summary")
    collection = store.collection(manifest)
    result = collection.query(query_embeddings=[[1, 0, 0]], n_results=3)
    assert summary_id in result["ids"][0]
    before = store.pointer.read_bytes()
    monkeypatch.setattr(
        store, "validate", lambda *a, **k: (_ for _ in ()).throw(ValueError("validation failed"))
    )
    other = config.data / "builds" / "failed_staging"
    other.mkdir()
    with pytest.raises(ValueError, match="validation failed"):
        store.persist_and_promote("failed_staging", nodes, vectors, ingestion, build_contract, reason)
    assert store.pointer.read_bytes() == before and not (other / "COMPLETE").exists()
