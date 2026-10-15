from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Leaf(StrictModel):
    leaf_id: str
    doc_id: str
    schema_version: str = "1"
    source_path: str
    original_name: str
    source_sha256: str
    text: str
    embedding_text: str
    content_sha256: str
    language: str
    headings: list[str]
    section: str | None = None
    provenance: list[dict[str, Any]]
    pdf_pages: list[int]
    printed_page: str | None = None
    location: dict[str, Any]
    order: int
    token_policy: dict[str, Any]


class Node(StrictModel):
    node_id: str
    node_type: Literal["leaf", "summary"]
    level: int
    text: str
    embedding_text: str
    content_sha256: str
    child_node_ids: list[str] = Field(default_factory=list)
    parent_node_ids: list[str] = Field(default_factory=list)
    source_leaf_ids: list[str]
    doc_ids: list[str]
    source_paths: list[str]
    provenance: list[dict[str, Any]]
    language: str = "unknown"
    summary_metadata: dict[str, Any] = Field(default_factory=dict)
    embedding_fingerprint: str = ""


class Subquestion(StrictModel):
    id: int
    question: str = Field(min_length=5, max_length=900)
    depends_on: list[int]


class Plan(StrictModel):
    original_question: str
    subquestions: list[Subquestion] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def valid_sequence(self):
        for i, item in enumerate(self.subquestions, 1):
            if item.id != i or item.depends_on != list(range(1, i)) or not item.question.strip():
                raise ValueError("IDs e dependências devem ser 1:[], 2:[1], 3:[1,2]")
        texts = [" ".join(q.question.lower().split()) for q in self.subquestions]
        if len(set(texts)) != 3:
            raise ValueError("Subperguntas duplicadas")
        return self


class Fact(StrictModel):
    text: str = Field(min_length=1, max_length=500)
    evidence_ids: list[str] = Field(min_length=1, max_length=5)


class Hypothesis(StrictModel):
    text: str = Field(max_length=400)
    status: Literal["unverified"]


class StepAnswer(StrictModel):
    step_id: int
    answer: str = Field(max_length=1800)
    supported_facts: list[Fact] = Field(max_length=3)
    hypotheses: list[Hypothesis] = Field(max_length=2)
    missing_information: list[str] = Field(max_length=3)
    corrections: list[str] = Field(default_factory=list, max_length=2)
    evidence_ids: list[str] = Field(max_length=10)


class SummaryAnswer(StrictModel):
    text: str = Field(min_length=1, max_length=700)


class FinalAnswer(StrictModel):
    answer: str = Field(min_length=1)
    evidence_ids: list[str]
    limitations: list[str]
