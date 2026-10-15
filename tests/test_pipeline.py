import json
import pytest
from xunit_rag.schemas import FinalAnswer, Plan, StepAnswer
from xunit_rag.specialist import Specialist, load_prompts
from xunit_rag.tokenization import PromptBudget
from xunit_rag.utils import Trace, read_jsonl
from conftest import make_node


QUESTION = "Como distinguir Test Stub e Mock Object no PaymentService?"


def plan():
    return Plan(
        original_question="LLM reproduction is ignored",
        subquestions=[
            {
                "id": 1,
                "question": "Qual comportamento do PaymentService exige resposta controlada de Test Stub?",
                "depends_on": [],
            },
            {
                "id": 2,
                "question": "Como Mock Object verifica interações do PaymentService?",
                "depends_on": [1],
            },
            {
                "id": 3,
                "question": "Como escolher Test Stub ou Mock Object para testar PaymentService?",
                "depends_on": [1, 2],
            },
        ],
    )


class FakeSession:
    def __init__(self, events, invalid=0, fabricated=False):
        self.events, self.invalid, self.fabricated = events, invalid, fabricated
        self.calls = 0
        self.contexts = []

    def generate(self, role, system, user, schema, output_tokens, budget):
        self.calls += 1
        self.events.append(role)
        payload = json.loads(user)
        self.contexts.append((role, payload))
        if role.startswith("decomposition"):
            if self.invalid:
                self.invalid -= 1
                raise ValueError("invalid JSON from test transport")
            return plan()
        if role == "synthesis":
            return FinalAnswer(answer="Diagnóstico com fonte [l_1].", evidence_ids=["l_1"], limitations=[])
        i = payload["current_step"]
        node_id = "l_invented" if self.fabricated else f"l_{i}"
        return StepAnswer(
            step_id=i,
            answer="Resposta curta",
            supported_facts=[{"text": f"Fato {i}", "evidence_ids": [node_id]}],
            hypotheses=[{"text": "Hipótese explicitamente não verificada", "status": "unverified"}],
            missing_information=[],
            evidence_ids=[node_id],
        )


class FakeRetriever:
    def __init__(self, events, empty=False):
        self.events, self.calls, self.queries, self.empty = events, 0, [], empty

    def retrieve(self, question):
        self.calls += 1
        self.queries.append(question)
        self.events.append(f"retrieve{self.calls}")
        return [] if self.empty else [{"node": make_node(self.calls), "rank": 1, "cosine_distance": 0.1}]


def specialist(config, tokenizer, **kwargs):
    events = []
    session = FakeSession(events, **kwargs)
    retriever = FakeRetriever(events)
    trace = Trace(config.root / "logs" / "test.jsonl", "test")
    engine = Specialist(config, session, retriever, PromptBudget(config, tokenizer), load_prompts(), trace)
    return engine, session, retriever, trace, events


def test_strict_sequence_and_previous_evidence(config, tokenizer):
    engine, session, retriever, trace, events = specialist(config, tokenizer)
    result = engine.answer(QUESTION)
    assert result.evidence_ids == ["l_1"]
    assert events == [
        "decomposition",
        "retrieve1",
        "step_answer",
        "retrieve2",
        "step_answer",
        "retrieve3",
        "step_answer",
        "synthesis",
    ]
    assert session.calls == 5 and retriever.calls == 3
    contexts = [p for role, p in session.contexts if role == "step_answer"]
    assert [len(p["previous_results"]) for p in contexts] == [0, 1, 2]
    assert len(session.contexts[-1][1]["previous_results"]) == 3
    assert contexts[2]["previous_results"][0]["supported_facts"][0]["evidence_ids"] == ["l_1"]
    assert contexts[2]["previous_results"][0]["hypotheses"][0]["status"] == "unverified"
    assert retriever.queries == [q.question for q in plan().subquestions]
    assert all("Resposta curta" not in q for q in retriever.queries)
    assert read_jsonl(trace.path)[0]["plan"]["original_question"] == QUESTION


def test_one_decomposition_repair(config, tokenizer):
    engine, session, retriever, _, _ = specialist(config, tokenizer, invalid=1)
    engine.answer(QUESTION)
    assert session.calls == 6 and retriever.calls == 3


def test_invalid_plan_stops_without_retrieval(config, tokenizer):
    engine, session, retriever, _, _ = specialist(config, tokenizer, invalid=2)
    with pytest.raises(ValueError, match="único reparo"):
        engine.answer(QUESTION)
    assert session.calls == 2 and retriever.calls == 0


def test_fabricated_evidence_rejected(config, tokenizer):
    engine, _, retriever, _, _ = specialist(config, tokenizer, fabricated=True)
    with pytest.raises(ValueError, match="Citações inexistentes"):
        engine.answer(QUESTION)
    assert retriever.calls == 1


@pytest.mark.parametrize("mutation", ["count", "depends", "duplicate", "empty"])
def test_plan_validation(mutation):
    value = plan().model_dump()
    if mutation == "count":
        value["subquestions"].pop()
    elif mutation == "depends":
        value["subquestions"][1]["depends_on"] = []
    elif mutation == "duplicate":
        value["subquestions"][1]["question"] = value["subquestions"][0]["question"]
    else:
        value["subquestions"][0]["question"] = ""
    with pytest.raises(ValueError):
        Plan.model_validate(value)


def test_budget_counts_prior_results_and_reserves(config, tokenizer):
    budget = PromptBudget(config, tokenizer)
    assert not budget.fits("rules " * 3000, "question " * 600, 600)
    assert budget.fits("rules", "question", 600)
    with pytest.raises(ValueError, match="num_ctx"):
        budget.require("rules", "prior_sources " * 4000, 600)


def test_empty_evidence_has_no_allowed_ids(config, tokenizer):
    engine, _, _, _, _ = specialist(config, tokenizer)
    _, user, allowed = engine.make_context("step_answer", QUESTION, [], [], 600, plan().subquestions[0])
    assert not allowed and json.loads(user)["evidence"] == []


def test_huge_user_code_not_silently_truncated(config, tokenizer):
    engine, _, _, _, _ = specialist(config, tokenizer)
    with pytest.raises(ValueError, match="num_ctx"):
        engine.answer("PaymentService " * 5000)


def test_current_retrieval_gets_budget_while_prior_facts_keep_source_ids(config, tokenizer):
    from dataclasses import replace

    config = replace(config, num_ctx=2000)
    engine, _, _, trace, _ = specialist(config, tokenizer)
    old, current = make_node(1, "oldsource " * 700), make_node(2, "newsource " * 450)
    engine.registry.register([{"node": old}, {"node": current}])
    engine.registry.sent([old.node_id])
    prior = StepAnswer(step_id=1, answer="Prior result", supported_facts=[{
        "text": "Previously established fact", "evidence_ids": [old.node_id]
    }], hypotheses=[], missing_information=[], evidence_ids=[old.node_id])
    _, user, allowed = engine.make_context(
        "step_answer", QUESTION, [prior], [{"node": current}], 600, plan().subquestions[1]
    )
    payload = json.loads(user)
    assert [e["node_id"] for e in payload["evidence"]] == [current.node_id]
    assert payload["previous_results"][0]["supported_facts"][0]["evidence_ids"] == [old.node_id]
    assert payload["evidence_omitted_for_budget"][0]["node_id"] == old.node_id
    assert allowed == {old.node_id, current.node_id}
    assert "oldsource" not in user and "newsource" in user
    assert read_jsonl(trace.path)[-1]["budget"]["input_tokens"] + 760 <= config.num_ctx


def test_source_lists_derive_from_validated_facts_and_citations(config, tokenizer):
    engine, session, _, trace, _ = specialist(config, tokenizer)
    engine.registry.register([{"node": make_node(1)}, {"node": make_node(2)}])
    engine.registry.sent(["l_1", "l_2"])
    step = StepAnswer(step_id=1, answer="A supported fact", supported_facts=[{
        "text": "Fact", "evidence_ids": ["l_1"]
    }], hypotheses=[], missing_information=[], evidence_ids=[])
    engine.validate_step_sources(step, {"l_1", "l_2"})
    assert step.evidence_ids == ["l_1"]
    final = FinalAnswer(answer="Claim [node_id:l_1].", evidence_ids=["l_2"], limitations=[])
    engine.validate_final_sources(final, {"l_1", "l_2"})
    assert final.answer == "Claim [l_1]." and final.evidence_ids == ["l_1"]
    assert session.calls == 0
    assert [event["event"] for event in read_jsonl(trace.path)] == [
        "source_list_completion", "citation_normalization"
    ]


@pytest.mark.parametrize("answer,declared", [
    ("Claim [node_id:l_fake].", ["l_1"]),
    ("Claim [l_1].", ["l_fake"]),
    ("Claim [node_id:l_fake-id].", []),
])
def test_citation_formatting_never_hides_unknown_or_missing_sources(config, tokenizer, answer, declared):
    engine, _, _, _, _ = specialist(config, tokenizer)
    engine.registry.register([{"node": make_node(1)}])
    engine.registry.sent(["l_1"])
    with pytest.raises(ValueError):
        engine.validate_final_sources(FinalAnswer(answer=answer, evidence_ids=declared, limitations=[]), {"l_1"})


def test_validated_general_references_are_displayed_without_claim_level_attribution(config, tokenizer):
    engine, session, _, trace, _ = specialist(config, tokenizer)
    engine.registry.register([{"node": make_node(1)}])
    engine.registry.sent(["l_1"])
    final = FinalAnswer(answer="A response with general references.", evidence_ids=["l_1"], limitations=[])
    engine.validate_final_sources(final, {"l_1"})
    assert final.answer == "A response with general references.\n\nFontes: [l_1]"
    assert final.evidence_ids == ["l_1"] and session.calls == 0
    assert read_jsonl(trace.path)[-1]["source_list_only"] is True
