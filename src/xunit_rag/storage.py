import importlib.metadata
import os
from pathlib import Path

import numpy as np
from filelock import FileLock

from .ingestion import scan_docs
from .schemas import Node
from .utils import atomic_json, canonical, digest, file_sha, read_json, read_jsonl, utc_now, write_jsonl

DEPENDENCIES = (
    "chromadb",
    "docling",
    "docling-core",
    "langchain-core",
    "httpx",
    "pydantic",
    "numpy",
    "scikit-learn",
    "umap-learn",
    "tokenizers",
    "semchunk",
    "pypdf",
)


def dependency_versions():
    return {name: importlib.metadata.version(name) for name in DEPENDENCIES}


def storage_path_policy():
    return "windows_native_ascii_alias_v1" if os.name == "nt" else "absolute_native_v1"


def native_chroma_path(path):
    path = path.resolve()
    if os.name != "nt" or str(path).isascii():
        return str(path)
    # Native HNSW file I/O on Windows cannot reliably reopen this installation's Unicode paths.
    # A verified DOS alias addresses the same directory; no database is moved or copied.
    import ctypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    get_short = kernel.GetShortPathNameW
    get_short.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
    get_short.restype = ctypes.c_uint
    buffer = ctypes.create_unicode_buffer(32768)
    if not get_short(str(path), buffer, len(buffer)):
        error = ctypes.get_last_error()
        raise OSError(
            error,
            "Não foi possível resolver o caminho nativo do Chroma. Execute fora do sandbox restritivo",
            str(path),
        )
    alias = buffer.value
    if not alias.isascii() or not Path(alias).samefile(path):
        raise ValueError(
            "Chroma precisa de alias ASCII para a mesma pasta data/chroma; nenhum diretório será movido"
        )
    return alias


def contract(config, embeddings, session, prompts, budget):
    return {
        "schema_version": "1",
        "storage_path_policy": storage_path_policy(),
        "profile": config.profile(),
        "dependencies": dependency_versions(),
        "embedding_policy": embeddings.policy,
        "embedding_fingerprint": embeddings.fingerprint,
        "models": {
            tag: {"digest": value["digest"], "template_sha256": value["template_sha256"]}
            for tag, value in session.model_info["models"].items()
        },
        "prompt_hashes": {k: digest(v) for k, v in prompts.items()},
        "budget_tokenizer": budget.tokenizer.policy,
        "query_translation": "query_decomposition",
        "decomposition_steps": 3,
        "indexing": "raptor",
        "vector_store": "chromadb",
        "retrieval": "collapsed_tree",
    }


def validate_graph(nodes, expected_leaf_ids=None):
    by_id = {n.node_id: n for n in nodes}
    if len(by_id) != len(nodes):
        raise ValueError("IDs duplicados no graph")
    leaves = {n.node_id for n in nodes if n.node_type == "leaf"}
    if expected_leaf_ids is not None and leaves != set(expected_leaf_ids):
        raise ValueError("Leaves divergentes do corpus canônico")
    for node in nodes:
        if digest(node.text) != node.content_sha256 or not node.provenance:
            raise ValueError(f"Conteúdo/proveniência inválida: {node.node_id}")
        if set(node.source_leaf_ids) - leaves:
            raise ValueError("source_leaf_ids inexistentes")
        if node.node_type == "leaf":
            if node.level != 0 or node.child_node_ids or node.source_leaf_ids != [node.node_id]:
                raise ValueError("Leaf inconsistente")
        else:
            if not node.child_node_ids or node.level < 1:
                raise ValueError("Summary sem children")
            union, doc_ids, source_paths, provenance = set(), set(), set(), set()
            for child_id in node.child_node_ids:
                child = by_id.get(child_id)
                if (
                    child is None
                    or child.level != node.level - 1
                    or node.node_id not in child.parent_node_ids
                ):
                    raise ValueError("Aresta inválida/ciclo ou nível incorreto")
                union.update(child.source_leaf_ids)
                doc_ids.update(child.doc_ids)
                source_paths.update(child.source_paths)
                provenance.update(canonical(p) for p in child.provenance)
            if union != set(node.source_leaf_ids):
                raise ValueError("Perda de proveniência na propagação")
            if doc_ids != set(node.doc_ids) or source_paths != set(node.source_paths):
                raise ValueError("Documentos/paths divergentes dos children")
            if provenance != {canonical(p) for p in node.provenance}:
                raise ValueError("Proveniência do summary divergente dos children")
        for parent_id in node.parent_node_ids:
            parent = by_id.get(parent_id)
            if parent is None or node.node_id not in parent.child_node_ids or parent.level != node.level + 1:
                raise ValueError("Parent inválido")
    summaries = [n for n in nodes if n.node_type == "summary"]
    if len(leaves) > 1:
        if not summaries or set().union(*(set(n.source_leaf_ids) for n in summaries)) != leaves:
            raise ValueError("RAPTOR incompleto: leaves sem summary")
    return {
        "leaves": len(leaves),
        "summaries": len(summaries),
        "summary_levels": max((n.level for n in nodes), default=0),
        "nodes": len(nodes),
    }


class IndexStore:
    def __init__(self, config):
        import chromadb
        from chromadb.config import Settings

        self.config = config
        self.client = chromadb.PersistentClient(
            path=native_chroma_path(config.data / "chroma"), settings=Settings(anonymized_telemetry=False)
        )
        self.pointer = config.data / "manifests" / "active.json"

    def active(self):
        pointer = read_json(self.pointer)
        if not pointer:
            raise ValueError("Índice ausente. Execute prepare.")
        if pointer.get("status") == "unavailable":
            raise ValueError(
                "Sem documentos suportados no índice ativo. Adicione fontes em docs e execute prepare."
            )
        directory = self.config.data / "builds" / pointer["build_id"]
        manifest = read_json(directory / "index_manifest.json")
        if not manifest or manifest.get("status") != "complete" or not (directory / "COMPLETE").exists():
            raise ValueError("Índice incompleto. Execute prepare.")
        if digest(manifest) != pointer["manifest_hash"]:
            raise ValueError("Manifest ativo diverge do indicador atômico. Execute prepare.")
        return directory, manifest

    def require_current(self, expected_contract=None, model_info=None):
        directory, manifest = self.active()
        if digest(scan_docs(self.config)) != manifest["inventory_hash"]:
            raise ValueError("docs mudou. Execute prepare antes de consultar.")
        if manifest["contract"]["profile"] != self.config.profile():
            raise ValueError("Perfil incompatível. Execute prepare.")
        if manifest["contract"]["dependencies"] != dependency_versions():
            raise ValueError("Versões de dependências mudaram. Execute prepare.")
        if manifest["contract"].get("storage_path_policy") != storage_path_policy():
            raise ValueError("Política de caminho nativo do Chroma mudou. Execute prepare.")
        if expected_contract and digest(expected_contract) != manifest["contract_hash"]:
            raise ValueError("Modelo/digest/prompt/tokenização incompatível. Execute prepare.")
        if model_info:
            for tag, model in manifest["contract"]["models"].items():
                if model_info["models"][tag]["digest"] != model["digest"]:
                    raise ValueError(f"Digest incompatível: {tag}. Execute prepare.")
        return directory, manifest

    def collection(self, manifest):
        return self.client.get_collection(manifest["collection"], embedding_function=None)

    def persist_and_promote(self, build_id, nodes, vectors, ingestion, build_contract, stop_reason):
        directory = self.config.data / "builds" / build_id
        counts = validate_graph(nodes)
        collection_name = "xunit_testing_specialist_raptor_" + build_id[:24]
        collection = self.client.get_or_create_collection(
            collection_name, embedding_function=None, configuration={"hnsw": {"space": "cosine"}}
        )
        write_jsonl(directory / "tree.jsonl", nodes)
        for offset in range(0, len(nodes), 100):
            batch = nodes[offset : offset + 100]
            collection.upsert(
                ids=[n.node_id for n in batch],
                embeddings=[vectors[n.node_id] for n in batch],
                documents=[n.text for n in batch],
                metadatas=[
                    {
                        "node_type": n.node_type,
                        "level": n.level,
                        "source_leaf_ids": canonical(n.source_leaf_ids),
                        "source_paths": canonical(n.source_paths),
                        "doc_ids": canonical(n.doc_ids),
                        "content_sha256": n.content_sha256,
                        "embedding_fingerprint": n.embedding_fingerprint,
                    }
                    for n in batch
                ],
            )
        manifest = {
            "status": "staging",
            "created_at": utc_now(),
            "build_id": build_id,
            "collection": collection_name,
            "counts": counts,
            "stop_reason": stop_reason,
            "inventory_hash": ingestion["inventory_hash"],
            "corpus_hash": ingestion["corpus_hash"],
            "document_count": sum(v["status"] in ("ingested", "reused") for v in ingestion["files"].values()),
            "contract": build_contract,
            "contract_hash": digest(build_contract),
            "tree_sha256": file_sha(directory / "tree.jsonl"),
        }
        atomic_json(directory / "index_manifest.json", manifest)
        self.validate(directory, manifest, allow_staging=True)
        # A source mutation during a long build cannot promote a stale index.
        if digest(scan_docs(self.config)) != ingestion["inventory_hash"]:
            raise ValueError("docs mudou durante o build. Execute prepare novamente.")
        manifest["status"] = "complete"
        atomic_json(directory / "index_manifest.json", manifest)
        atomic_json(directory / "COMPLETE", {"manifest_hash": digest(manifest)})
        atomic_json(self.pointer, {"build_id": build_id, "manifest_hash": digest(manifest)})
        atomic_json(self.config.data / "manifests" / "index_manifest.json", manifest)
        return manifest

    def validate(self, directory=None, manifest=None, allow_staging=False):
        if directory is None:
            directory, manifest = self.require_current()
        if not allow_staging and manifest["status"] != "complete":
            raise ValueError("Índice incompleto")
        tree_path = directory / "tree.jsonl"
        if file_sha(tree_path) != manifest["tree_sha256"]:
            raise ValueError("Hash tree divergente")
        nodes = [Node.model_validate(v) for v in read_jsonl(tree_path)]
        corpus = read_jsonl(self.config.data / "processed" / "ingest_corpus.jsonl")
        if digest(corpus) != manifest["corpus_hash"]:
            raise ValueError("Hash corpus divergente")
        counts = validate_graph(nodes, [v["leaf_id"] for v in corpus])
        by_leaf = {leaf["leaf_id"]: leaf for leaf in corpus}
        for node in nodes:
            if node.node_type == "leaf":
                leaf = by_leaf[node.node_id]
                if node.text != leaf["text"] or node.embedding_text != leaf["embedding_text"]:
                    raise ValueError("Leaf modificado fora da ingestão canônica")
                if node.source_paths != [leaf["source_path"]] or node.doc_ids != [leaf["doc_id"]]:
                    raise ValueError("Origem do leaf divergente do corpus")
                for p in node.provenance:
                    if (
                        p.get("leaf_id") != leaf["leaf_id"]
                        or p.get("source_path") != leaf["source_path"]
                        or p.get("pdf_pages") != leaf["pdf_pages"]
                        or p.get("location") != leaf["location"]
                        or p.get("docling") != leaf["provenance"]
                    ):
                        raise ValueError("Proveniência do leaf divergente do corpus")
        if counts != manifest["counts"]:
            raise ValueError("Contagens divergentes")
        collection = self.collection(manifest)
        if collection.count() != len(nodes):
            raise ValueError("Contagem Chroma divergente")
        if collection.configuration.get("hnsw", {}).get("space") != "cosine":
            raise ValueError("Métrica Chroma incompatível")
        by_id = {n.node_id: n for n in nodes}
        dimension = manifest["contract"]["embedding_policy"]["dimension"]
        fingerprint = manifest["contract"]["embedding_fingerprint"]
        observed = set()
        for offset in range(0, len(nodes), 100):
            data = collection.get(limit=100, offset=offset, include=["embeddings", "metadatas", "documents"])
            for i, node_id in enumerate(data["ids"]):
                if node_id not in by_id or node_id in observed:
                    raise ValueError("ID Chroma inválido/duplicado")
                observed.add(node_id)
                node, metadata = by_id[node_id], data["metadatas"][i]
                vector = np.asarray(data["embeddings"][i])
                if (
                    vector.shape != (dimension,)
                    or not np.isfinite(vector).all()
                    or not np.isclose(np.linalg.norm(vector), 1, atol=1e-4)
                ):
                    raise ValueError("Vetor Chroma inválido")
                if (
                    metadata["embedding_fingerprint"] != fingerprint
                    or node.embedding_fingerprint != fingerprint
                ):
                    raise ValueError("Espaços vetoriais incompatíveis")
                if data["documents"][i] != node.text or metadata["content_sha256"] != node.content_sha256:
                    raise ValueError("Texto/metadata Chroma divergente")
        return {"status": "valid", **counts, "dimension": dimension, "build_id": manifest["build_id"]}


def build_lock(config):
    return FileLock(str(config.data / "prepare.lock"), timeout=0)
