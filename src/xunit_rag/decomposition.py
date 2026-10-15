import re

from pydantic import ValidationError

from .schemas import Plan
from .utils import canonical


def validate_relation(plan, original_question):
    # A transparent structural relevance check, not a semantic quality or truth score.
    original_terms = {v.casefold() for v in re.findall(r"\w{4,}", original_question)}
    generic = {
        "como",
        "para",
        "what",
        "this",
        "that",
        "with",
        "qual",
        "quais",
        "does",
        "teste",
        "test",
        "testes",
    }
    original_terms -= generic
    combined = {v.casefold() for q in plan.subquestions for v in re.findall(r"\w{4,}", q.question)}
    if original_terms and not original_terms & combined:
        raise ValueError(
            "Plano sem termos ligados à pergunta original; relação semântica precisa de revisão humana"
        )
    for q in plan.subquestions:
        if re.search(
            r"\b(esse caso|este caso|o padrão acima|this case|the above pattern)\b", q.question, re.IGNORECASE
        ):
            raise ValueError("Subpergunta contém referência não autossuficiente")
    plan.original_question = original_question
    return plan


def decompose(session, budget, prompt, question, trace):
    user = canonical({"original_question": question})
    for attempt in range(2):
        try:
            plan = session.generate(
                "decomposition" if attempt == 0 else "decomposition_repair", prompt, user, Plan, 600, budget
            )
            plan = validate_relation(plan, question)
            trace.emit(
                "plan",
                plan=plan.model_dump(),
                repaired=bool(attempt),
                validation="structural; semantic coverage requires human review",
            )
            return plan
        except (ValidationError, ValueError) as error:
            trace.emit("plan_invalid", attempt=attempt + 1, error=str(error))
            if attempt:
                raise ValueError("Decomposição inválida após o único reparo permitido") from error
            user = canonical(
                {
                    "original_question": question,
                    "repair": "Return a corrected complete three-step JSON plan",
                    "validation_error": str(error)[:1200],
                }
            )
    raise AssertionError("Reparo esgotado")
