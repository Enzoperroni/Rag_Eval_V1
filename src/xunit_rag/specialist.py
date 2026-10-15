import re
import time
import uuid
from pathlib import Path

from .decomposition import decompose
from .models import OllamaSession, QwenEmbeddings
from .retrieval import EvidenceRegistry, Retriever, evidence_payload
from .schemas import FinalAnswer, StepAnswer
from .storage import IndexStore, contract
from .tokenization import PromptBudget, QwenTokenizer
from .utils import Trace, canonical, digest, timed


def load_prompts():
    directory = Path(__file__).parent / "prompts"
    prompts = {
        name: (directory / f"{name}.txt").read_text(encoding="utf-8")
        for name in ("decomposition", "step_answer", "summary", "synthesis")
    }
    prompts["decision_rules"] = (directory / "decision_rules.md").read_text(encoding="utf-8")
    return prompts


def compact_step(answer):
    return {
        "step_id": answer.step_id,
        "supported_facts": [f.model_dump() for f in answer.supported_facts],
        "hypotheses": [h.model_dump() for h in answer.hypotheses],
        "missing_information": answer.missing_information,
        "corrections": answer.corrections,
    }


class Specialist:
    def __init__(self, config, session, retriever, budget, prompts, trace):
        self.config, self.session, self.retriever, self.budget = config, session, retriever, budget
        self.prompts, self.trace = prompts, trace
        self.registry = EvidenceRegistry()

    def system(self, role):
        return self.prompts[role] + "\n\n" + self.prompts["decision_rules"]

    def validate_step_sources(self, answer, allowed):
        self.registry.validate_ids(answer.evidence_ids, allowed)
        fact_ids = {node_id for fact in answer.supported_facts for node_id in fact.evidence_ids}
        self.registry.validate_ids(fact_ids, allowed)
        declared = set(answer.evidence_ids)
        if fact_ids - declared:
            self.trace.emit("source_list_completion", step_id=answer.step_id,
                            added_from_validated_facts=sorted(fact_ids - declared))
        answer.evidence_ids = sorted(declared | fact_ids)

    def validate_final_sources(self, final, allowed):
        self.registry.validate_ids(final.evidence_ids, allowed)
        normalized = re.sub(r"\[node_id:\s*((?:l|s)_[^\]]+)\]", r"[\1]", final.answer)
        cited = set(re.findall(r"\[((?:l|s)_[^\]]+)\]", normalized))
        self.registry.validate_ids(cited, allowed)
        source_list_only = bool(final.evidence_ids) and not cited
        if source_list_only:
            cited = set(final.evidence_ids)
            normalized += "\n\nFontes: " + ", ".join(f"[{node_id}]" for node_id in sorted(cited))
        if normalized != final.answer or cited != set(final.evidence_ids):
            self.trace.emit("citation_normalization", declared_ids=final.evidence_ids,
                            cited_ids=sorted(cited), source_list_only=source_list_only,
                            prefix_removed="[node_id:" in final.answer)
        final.answer, final.evidence_ids = normalized, sorted(cited)

    def make_context(self, role, question, previous, retrieved, output_tokens, subquestion=None):
        system = self.system(role)
        prior = [compact_step(answer) for answer in previous]
        prior_ids = {
            node_id for answer in previous for fact in answer.supported_facts for node_id in fact.evidence_ids
        }
        self.registry.validate_ids(prior_ids)
        base = {
            "original_question": question,
            "previous_results": prior,
            "evidence": [],
            "evidence_omitted_for_budget": [],
            "source_policy": "node IDs resolve through this run's EvidenceRegistry",
        }
        if subquestion:
            base.update(current_step=subquestion.id, subquestion=subquestion.question)
        self.budget.require(system, canonical(base), output_tokens)
        # Prior facts and their IDs are already in base. Give the current search room to add evidence,
        # then restore earlier full passages when they fit; omitted passages retain explicit metadata.
        candidates = []
        for item in retrieved:
            if item["node"].node_id not in {n.node_id for n in candidates}:
                candidates.append(item["node"])
        candidate_ids = {node.node_id for node in candidates}
        candidates.extend(self.registry.nodes[node_id] for node_id in sorted(prior_ids - candidate_ids))
        used, omitted = [], []
        for node in candidates:
            payload = evidence_payload(node)
            candidate = {**base, "evidence": used + [payload], "evidence_omitted_for_budget": omitted}
            if self.budget.fits(system, canonical(candidate), output_tokens):
                used.append(payload)
            else:
                omitted.append(
                    {
                        "node_id": node.node_id,
                        "type": node.node_type,
                        "level": node.level,
                        "reason": "whole evidence omitted for token budget; no text truncation",
                    }
                )
        base.update(evidence=used, evidence_omitted_for_budget=omitted)
        while not self.budget.fits(system, canonical(base), output_tokens) and used:
            removed = used.pop()
            omitted.append(
                {
                    "node_id": removed["node_id"],
                    "type": removed["node_type"],
                    "level": removed["level"],
                    "reason": "omission metadata budget",
                }
            )
        token_info = self.budget.require(system, canonical(base), output_tokens)
        used_ids = {p["node_id"] for p in used}
        self.registry.sent(used_ids)
        allowed_ids = used_ids | prior_ids
        self.trace.emit(
            "context",
            role=role,
            step_id=subquestion.id if subquestion else None,
            effective_input=base,
            evidence_sent=sorted(used_ids),
            evidence_omitted=omitted,
            previous_fact_evidence=sorted(prior_ids),
            budget=token_info,
        )
        return system, canonical(base), allowed_ids

    def answer(self, question):
        plan = decompose(self.session, self.budget, self.prompts["decomposition"], question, self.trace)
        answers = []
        for subquestion in plan.subquestions:
            with timed(self.trace, "step_time", step_id=subquestion.id):
                retrieved = self.retriever.retrieve(subquestion.question)
                self.registry.register(retrieved)
                system, user, allowed = self.make_context(
                    "step_answer", question, answers, retrieved, 600, subquestion
                )
                answer = self.session.generate("step_answer", system, user, StepAnswer, 600, self.budget)
                if answer.step_id != subquestion.id:
                    raise ValueError("LLM retornou step_id incorreto")
                self.validate_step_sources(answer, allowed)
                answers.append(answer)
                self.trace.emit("step_answer", answer=answer.model_dump())
        final_ids = list(dict.fromkeys(
            node_id for answer in reversed(answers)
            for fact in answer.supported_facts for node_id in fact.evidence_ids
        ))
        final_ids.extend(sorted(self.registry.sent_ids - set(final_ids)))
        final_candidates = [{"node": self.registry.nodes[node_id]} for node_id in final_ids]
        system, user, allowed = self.make_context("synthesis", question, answers, final_candidates, 1200)
        final = self.session.generate("synthesis", system, user, FinalAnswer, 1200, self.budget)
        self.validate_final_sources(final, allowed)
        self.trace.emit(
            "synthesis", result=final.model_dump(), sources=self.registry.payloads(final.evidence_ids)
        )
        return final


def ask(config, question):
    run_id = uuid.uuid4().hex
    trace = Trace(config.root / "logs" / f"{run_id}.jsonl", run_id)
    started = time.monotonic()
    session = OllamaSession(config, trace, deadline=started + config.query_deadline)
    trace.emit("query_start", original_question=question, profile_hash=digest(config.profile()))
    try:
        store = IndexStore(config)
        store.require_current()
        session.inspect_models()
        _, compatible_manifest = store.require_current(model_info=session.model_info)
        tokenizer = QwenTokenizer(config, "embedding")
        budget = PromptBudget(config, QwenTokenizer(config, "llm"))
        prompts = load_prompts()
        embeddings = QwenEmbeddings(
            config,
            session,
            tokenizer,
            session.model_info,
            measured_dimension=compatible_manifest["contract"]["embedding_policy"]["dimension"],
        )
        build_contract = contract(config, embeddings, session, prompts, budget)
        directory, manifest = store.require_current(expected_contract=build_contract)
        trace.emit(
            "query_index",
            corpus_hash=manifest["corpus_hash"],
            index_hash=manifest["contract_hash"],
            build_id=manifest["build_id"],
            models=session.model_info,
            embedding_policy=embeddings.policy,
            parameters=config.profile(),
        )
        retriever = Retriever(store, directory, manifest, embeddings, trace)
        final = Specialist(config, session, retriever, budget, prompts, trace).answer(question)
        session.close()
        trace.emit(
            "query_complete",
            status="success",
            total_seconds=time.monotonic() - started,
            logical_llm_calls=session.logical_calls,
            retrieval_calls=retriever.calls,
            transport_inference_attempts=session.transport_attempts,
        )
        return final, run_id
    except BaseException as error:
        try:
            session.close()
        finally:
            trace.emit(
                "query_error",
                status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                error=f"{type(error).__name__}: {error}",
                total_seconds=time.monotonic() - started,
                logical_llm_calls=session.logical_calls,
                transport_inference_attempts=session.transport_attempts,
            )
        raise
