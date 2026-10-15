import json
import numpy as np
import pytest
from xunit_rag.retrieval import EvidenceRegistry
from xunit_rag.tokenization import PromptBudget
from xunit_rag.utils import Trace
from xunit_rag.specialist import Specialist, load_prompts
from conftest import make_leaf, make_node
from test_raptor_storage import builder


def test_soft_threshold_and_argmax_fallback(config, tokenizer, monkeypatch):
    class GMM:
        def __init__(self, n_components, **kwargs):
            self.n_components = n_components

        def fit(self, vectors):
            return self

        def bic(self, vectors):
            return 0 if self.n_components == 2 else 100

        def predict_proba(self, vectors):
            return np.asarray([[0.5, 0.5], [0.99, 0.01], [0.01, 0.99], [0.09, 0.08]])

    monkeypatch.setattr("xunit_rag.raptor.GaussianMixture", GMM)
    tree = builder(config, tokenizer, config.data / "builds" / "soft_actual")
    nodes = [make_node(i) for i in range(4)]
    groups = tree.gmm_groups(nodes, np.asarray([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]]))
    assert sum(nodes[0] in group for group in groups) == 2
    assert sum(nodes[3] in group for group in groups) == 1
    assert {n.node_id for group in groups for n in group} == {n.node_id for n in nodes}


def test_summary_digest_invalidates_cache(config, tokenizer):
    first = builder(config, tokenizer, config.data / "builds" / "digest")
    first.build([make_leaf(1), make_leaf(2)])
    second = builder(config, tokenizer, config.data / "builds" / "digest")
    second.session.model_info["models"][config.llm_model]["digest"] = "changed_digest"
    second.build([make_leaf(1), make_leaf(2)])
    assert second.session.calls == 1


def test_oversized_cluster_without_progress_fails(config, tokenizer):
    tree = builder(config, tokenizer, config.data / "builds" / "oversized")
    nodes = [make_node(i, "word " * 1500) for i in range(3)]
    tree.vectors = {node.node_id: [1, 0, 0] for node in nodes}
    tree.gmm_groups = lambda nodes, vectors, forced_components=None: [nodes]
    with pytest.raises(ValueError, match="não progrediu"):
        tree.subdivide(nodes)


def test_evidence_retrieved_but_not_sent_cannot_be_cited():
    registry = EvidenceRegistry()
    registry.register([{"node": make_node(1)}])
    with pytest.raises(ValueError, match="não enviadas"):
        registry.validate_ids(["l_1"])
    registry.sent(["l_1"])
    registry.validate_ids(["l_1"])


def test_budget_omits_whole_evidence_and_preserves_id(config, tokenizer):
    trace = Trace(config.root / "logs" / "budget.jsonl", "budget")
    engine = Specialist(config, None, None, PromptBudget(config, tokenizer), load_prompts(), trace)
    huge = make_node(1, "sourceword " * 6000)
    retrieved = [{"node": huge}]
    engine.registry.register(retrieved)
    from xunit_rag.schemas import Subquestion

    _, user, allowed = engine.make_context(
        "step_answer",
        "PaymentService?",
        [],
        retrieved,
        600,
        Subquestion(id=1, question="PaymentService Test Stub?", depends_on=[]),
    )
    context = json.loads(user)
    assert context["evidence"] == [] and not allowed
    assert context["evidence_omitted_for_budget"][0]["node_id"] == huge.node_id
    assert "sourceword" not in user
