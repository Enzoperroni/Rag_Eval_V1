from .schemas import Node
from .utils import read_jsonl


class Retriever:
    def __init__(self, store, directory, manifest, embeddings, trace):
        self.collection, self.embeddings, self.trace = store.collection(manifest), embeddings, trace
        self.top_k = store.config.top_k
        self.nodes = {v["node_id"]: Node.model_validate(v) for v in read_jsonl(directory / "tree.jsonl")}
        self.calls = 0

    def retrieve(self, subquestion):
        vector = self.embeddings.embed_query(subquestion)
        self.calls += 1
        result = self.collection.query(
            query_embeddings=[vector], n_results=min(self.top_k, len(self.nodes)), include=["distances"]
        )
        pairs = sorted(zip(result["ids"][0], result["distances"][0]), key=lambda p: (p[1], p[0]))
        found, seen = [], set()
        for node_id, distance in pairs:
            if node_id in seen:
                continue
            seen.add(node_id)
            node = self.nodes[node_id]
            found.append({"node": node, "rank": len(found) + 1, "cosine_distance": float(distance)})
        self.trace.emit(
            "retrieval",
            query=subquestion,
            embedding_input=self.embeddings.query_input(subquestion),
            retrieval_call=self.calls,
            nodes=[
                {
                    **{k: v for k, v in item.items() if k != "node"},
                    "node_id": item["node"].node_id,
                    "type": item["node"].node_type,
                    "level": item["node"].level,
                    "provenance": item["node"].provenance,
                }
                for item in found
            ],
        )
        return found


def evidence_payload(node):
    # Summaries carry descendant source IDs, not a made-up single page attribution.
    navigation = node.node_type == "leaf" and any(
        heading.strip().casefold() in {"contents", "table of contents", "index", "sumário", "índice"}
        for item in node.provenance for heading in item.get("location", {}).get("headings", [])
    )
    return {
        "node_id": node.node_id,
        "node_type": node.node_type,
        "level": node.level,
        "text": node.text,
        "navigation_only": navigation,
        "source_paths": node.source_paths,
        "source_leaf_count": len(node.source_leaf_ids),
        "locations": [
            {
                "source_path": p["source_path"],
                "pdf_pages": p.get("pdf_pages", []),
                "printed_page": p.get("printed_page"),
                "headings": p.get("location", {}).get("headings", []),
            }
            for p in node.provenance
        ]
        if node.node_type == "leaf"
        else [],
        "source_notice": "generated RAPTOR summary; pages do not establish each claim"
        if node.node_type == "summary"
        else "navigation passage: names/locations only, not support for behavioral claims or recommendations"
        if navigation else "original source passage",
    }


class EvidenceRegistry:
    def __init__(self):
        self.nodes = {}
        self.sent_ids = set()

    def register(self, retrieved):
        for item in retrieved:
            node = item["node"]
            self.nodes[node.node_id] = node

    def sent(self, ids):
        if set(ids) - self.nodes.keys():
            raise ValueError("Evidência não recuperada")
        self.sent_ids.update(ids)

    def validate_ids(self, ids, allowed=None):
        available = self.sent_ids if allowed is None else set(allowed) & self.sent_ids
        unknown = set(ids) - available
        if unknown:
            raise ValueError(f"Citações inexistentes/não enviadas: {sorted(unknown)}")

    def payloads(self, ids):
        return [evidence_payload(self.nodes[node_id]) for node_id in sorted(set(ids))]
