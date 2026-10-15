import argparse
import json
import sys
import uuid

from .config import Config
from .ingestion import DocumentIngestor, scan_docs
from .models import OllamaSession, QwenEmbeddings
from .raptor import RaptorBuilder
from .specialist import ask, load_prompts
from .storage import IndexStore, build_lock, contract, dependency_versions
from .tokenization import PromptBudget, QwenTokenizer
from .utils import Trace, atomic_json, digest, read_jsonl


def print_json(value):
    print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)


def render_answer(config, final, run_id):
    print(final.answer)
    if final.limitations:
        print("\nLimitações: " + "; ".join(final.limitations))
    events = read_jsonl(config.root / "logs" / f"{run_id}.jsonl")
    synthesis = next(event for event in events if event["event"] == "synthesis")
    if synthesis["sources"]:
        print("\nFontes:")
        for source in synthesis["sources"]:
            paths = "; ".join(source["source_paths"])
            if source["node_type"] == "summary":
                location = f"summary RAPTOR gerado; {source['source_leaf_count']} leaves de origem"
            else:
                pages = sorted({page for item in source["locations"] for page in item["pdf_pages"]})
                location = (
                    "páginas PDF " + ", ".join(map(str, pages)) if pages else "refs estruturais no trace"
                )
            print(f"- [{source['node_id']}] {paths} — {location}")
    print(f"\nTrace: {run_id}")


def doctor(config):
    trace = Trace(config.root / "logs" / f"doctor_{uuid.uuid4().hex}.jsonl", "doctor")
    session = OllamaSession(config, trace)
    report = {
        "python": sys.version,
        "executable": sys.executable,
        "venv": sys.prefix != sys.base_prefix,
        "root": str(config.root),
        "dependencies": dependency_versions(),
        "documents": scan_docs(config),
        "ollama_server_recommended": {"OLLAMA_NUM_PARALLEL": "1", "OLLAMA_MAX_LOADED_MODELS": "1"},
        "server_environment_observable": False,
    }
    try:
        report["ollama"] = session.inspect_models()
        tokenizer = QwenTokenizer(config)
        embeddings = QwenEmbeddings(config, session, tokenizer, session.model_info)
        report["embedding"] = {"dimension_measured": embeddings.dimension, "policy": embeddings.policy}
        budget = PromptBudget(config, QwenTokenizer(config, "llm"))
        store = IndexStore(config)
        try:
            _, manifest = store.require_current(contract(config, embeddings, session, load_prompts(), budget))
            report["index"] = {
                "status": "compatible",
                "counts": manifest["counts"],
                "build_id": manifest["build_id"],
            }
        except ValueError as error:
            report["index"] = {"status": "not_ready", "reason": str(error)}
        report["residency_observed"] = session.request("/api/ps")
        atomic_json(config.data / "manifests" / "doctor.json", report)
        return report
    finally:
        session.close()


def prepare(config):
    run_id = "prepare_" + uuid.uuid4().hex
    trace = Trace(config.root / "logs" / f"{run_id}.jsonl", run_id)
    session = OllamaSession(config, trace)
    try:
        with build_lock(config):
            trace.emit("prepare_start", inventory=scan_docs(config))
            tokenizer = QwenTokenizer(config, "embedding")
            ingestor = DocumentIngestor(config, tokenizer, progress=lambda msg: print(msg, flush=True))
            leaves, ingestion = ingestor.ingest()
            if not leaves:
                atomic_json(
                    config.data / "manifests" / "active.json",
                    {
                        "status": "unavailable",
                        "reason": "no_supported_documents",
                        "inventory_hash": ingestion["inventory_hash"],
                    },
                )
                result = {
                    "status": "no_documents",
                    "message": "Não há documentos suportados para ingerir; nenhum índice criado.",
                    "inventory": ingestion["files"],
                }
                trace.emit("prepare_empty", **result)
                return result
            session.inspect_models()
            embeddings = QwenEmbeddings(config, session, tokenizer, session.model_info)
            budget = PromptBudget(config, QwenTokenizer(config, "llm"))
            prompts = load_prompts()
            build_contract = contract(config, embeddings, session, prompts, budget)
            store = IndexStore(config)
            try:
                directory, manifest = store.require_current(build_contract)
                if manifest["corpus_hash"] == ingestion["corpus_hash"]:
                    checked = store.validate(directory, manifest)
                    trace.emit("prepare_reused", result=checked)
                    return {**checked, "status": "reused"}
            except ValueError:
                pass
            build_id = digest(
                {
                    "corpus": ingestion["corpus_hash"],
                    "contract": build_contract,
                    "inventory": ingestion["inventory_hash"],
                }
            )
            directory = config.data / "builds" / build_id
            directory.mkdir(parents=True, exist_ok=True)
            atomic_json(
                directory / "build_request.json", {"contract": build_contract, "ingestion": ingestion}
            )
            nodes, vectors, stop_reason = RaptorBuilder(
                config,
                embeddings,
                session,
                budget,
                prompts["summary"],
                directory,
                progress=lambda msg: print(msg, flush=True),
            ).build(leaves)
            manifest = store.persist_and_promote(
                build_id, nodes, vectors, ingestion, build_contract, stop_reason
            )
            checked = store.validate()
            trace.emit("prepare_complete", manifest=manifest, validation=checked)
            return {**checked, "status": "complete"}
    except BaseException as error:
        trace.emit(
            "prepare_error",
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        session.close()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    parser = argparse.ArgumentParser(
        description="xUnit local: Query Decomposition (3 etapas) + RAPTOR + ChromaDB"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("doctor", "prepare", "validate-index", "chat"):
        sub.add_parser(name)
    sub.add_parser("ask").add_argument("question")
    sub.add_parser("inspect").add_argument("--run-id", required=True)
    sub.add_parser("evaluate").add_argument("--cases", required=True)
    args = parser.parse_args()
    config = Config.load()
    config.initialize()
    try:
        if args.command == "doctor":
            print_json(doctor(config))
        elif args.command == "prepare":
            print_json(prepare(config))
        elif args.command == "validate-index":
            print_json(IndexStore(config).validate())
        elif args.command == "ask":
            final, run_id = ask(config, args.question)
            render_answer(config, final, run_id)
        elif args.command == "chat":
            IndexStore(config).require_current()
            print(
                "xUnit specialist — sair/exit para encerrar. Cada pergunta executa três etapas e uma síntese."
            )
            while True:
                question = input("\nVocê: ").strip()
                if question.casefold() in {"sair", "exit"}:
                    break
                if question:
                    try:
                        answer, run_id = ask(config, question)
                        render_answer(config, answer, run_id)
                    except Exception as error:
                        print(f"Erro: {error}", file=sys.stderr)
        elif args.command == "inspect":
            if not args.run_id or any(
                c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                for c in args.run_id
            ):
                raise ValueError("run-id inválido")
            path = config.root / "logs" / f"{args.run_id}.jsonl"
            if not path.exists():
                raise ValueError(f"Trace inexistente: {args.run_id}")
            print_json(read_jsonl(path))
        elif args.command == "evaluate":
            from .evaluation import evaluate

            print_json(evaluate(config, args.cases))
    except KeyboardInterrupt:
        print("\nInterrompido. Checkpoints e trace preservados.", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f"Erro: {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
