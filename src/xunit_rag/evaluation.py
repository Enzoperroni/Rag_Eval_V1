import time
import uuid
from pathlib import Path

from .specialist import ask
from .utils import atomic_json, read_json, read_jsonl


def evaluate(config, cases_path):
    cases_path = Path(cases_path)
    cases = read_jsonl(cases_path) if cases_path.suffix == ".jsonl" else read_json(cases_path)
    if not isinstance(cases, list) or not cases:
        raise ValueError("Cases deve ser uma lista JSON/JSONL não vazia de {id, question}")
    run_id = "evaluation_" + uuid.uuid4().hex
    destination = config.root / "eval" / f"{run_id}.json"
    results = []
    for index, case in enumerate(cases):
        started = time.monotonic()
        result = {"case_id": case.get("id", str(index)), "question": case["question"]}
        try:
            answer, query_run_id = ask(config, case["question"])
            result.update(status="success", run_id=query_run_id, answer=answer.model_dump())
        except Exception as error:
            result.update(status="failed", error=f"{type(error).__name__}: {error}")
        result["seconds"] = time.monotonic() - started
        result["human_review"] = {
            "decomposition": None,
            "retrieval": None,
            "conceptual_faithfulness": None,
            "code_usefulness": None,
        }
        results.append(result)
        atomic_json(
            destination,
            {
                "evaluation_id": run_id,
                "cases": results,
                "note": "Structural success is not semantic correctness. Separate human review required; no lexical score.",
            },
        )
    return {
        "path": str(destination),
        "success": sum(r["status"] == "success" for r in results),
        "cases": len(results),
    }
