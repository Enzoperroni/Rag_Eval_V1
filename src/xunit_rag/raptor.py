"""RAPTOR: UMAP, GMM/BIC, soft memberships, budgeted semantic subdivision, recursive summaries."""

import warnings

import numpy as np
from sklearn.mixture import GaussianMixture

from .config import PROMPT_VERSION
from .schemas import Node, SummaryAnswer
from .utils import atomic_json, canonical, digest, read_json, write_jsonl


class RaptorBuilder:
    def __init__(self, config, embeddings, session, budget, summary_prompt, build_dir, progress=print):
        self.config, self.embeddings, self.session, self.budget = config, embeddings, session, budget
        self.summary_prompt, self.build_dir, self.progress = summary_prompt, build_dir, progress
        self.build_dir.mkdir(parents=True, exist_ok=True)
        self.nodes = {}
        self.vectors = {}
        self.cluster_reports = []

    def reduce(self, vectors):
        n = len(vectors)
        if n < 5:
            return vectors  # documented terminal/small-N GMM path of the same builder
        from umap import UMAP

        dimension = min(self.config.reduction_dim, n - 2, vectors.shape[1])
        return UMAP(
            n_components=dimension,
            n_neighbors=min(15, n - 1),
            metric="cosine",
            random_state=self.config.seed,
            n_jobs=1,
            init="random",
        ).fit_transform(vectors)

    def gmm_groups(self, nodes, vectors, forced_components=None):
        n = len(nodes)
        if n <= 2 and forced_components is None:
            return [nodes]
        reduced = self.reduce(np.asarray(vectors))
        upper = min(self.config.max_clusters, n - 1)
        candidates = [forced_components] if forced_components else range(1, upper + 1)
        best, best_bic = None, float("inf")
        failures = []
        for k in candidates:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    model = GaussianMixture(
                        n_components=k,
                        covariance_type="diag",
                        reg_covar=1e-5,
                        random_state=self.config.seed,
                        n_init=1,
                        max_iter=200,
                    ).fit(reduced)
                bic = model.bic(reduced)
                if np.isfinite(bic) and bic < best_bic:
                    best, best_bic = model, bic
            except (ValueError, RuntimeWarning, np.linalg.LinAlgError) as error:
                failures.append({"k": k, "error": str(error)})
        if best is None:
            raise ValueError(f"GMM sem solução numérica: {failures}")
        probabilities = best.predict_proba(reduced)
        if not np.isfinite(probabilities).all():
            raise ValueError("GMM produziu probabilidades não finitas")
        groups = [[] for _ in range(best.n_components)]
        for index, node in enumerate(nodes):
            memberships = np.flatnonzero(probabilities[index] > self.config.threshold).tolist()
            if not memberships:
                memberships = [int(np.argmax(probabilities[index]))]
            for k in memberships:
                groups[k].append(node)
        self.cluster_reports.append(
            {
                "n": n,
                "components": best.n_components,
                "bic": float(best_bic),
                "forced": forced_components,
                "numerical_failures": failures,
            }
        )
        return unique_groups(groups)

    def cluster_input(self, children):
        return canonical({"children": [{"id": n.node_id, "text": n.embedding_text} for n in children]})

    def fits_cluster(self, children):
        user = self.cluster_input(children)
        return self.budget.tokenizer.count_tokens(user) <= self.config.cluster_tokens and self.budget.fits(
            self.summary_prompt, user, self.config.summary_tokens
        )

    def subdivide(self, group, depth=0):
        if self.fits_cluster(group):
            return [group]
        if len(group) <= 1 or depth >= 16:
            raise ValueError("Cluster excede orçamento sem subdivisão possível; checkpoint preservado")
        vectors = np.asarray([self.vectors[n.node_id] for n in group])
        subgroups = self.gmm_groups(group, vectors, forced_components=2)
        # Soft overlap that keeps the entire oversized group gives no progress: do not drop children.
        if len(subgroups) < 2 or any(len(g) >= len(group) for g in subgroups):
            raise ValueError("Subdivisão semântica não progrediu dentro do orçamento; sem truncamento")
        result = []
        for subgroup in subgroups:
            result.extend(self.subdivide(subgroup, depth + 1))
        return unique_groups(result)

    def summary(self, children, level):
        ids = sorted(n.node_id for n in children)
        llm = self.session.model_info["models"][self.config.llm_model]
        metadata = {
            "model": self.config.llm_model,
            "digest": llm["digest"],
            "prompt_version": PROMPT_VERSION,
            "prompt_hash": digest(self.summary_prompt),
            "temperature": self.config.temperature,
            "seed": self.config.seed,
            "num_ctx": self.config.num_ctx,
            "max_output_tokens": self.config.summary_tokens,
            "think": False,
        }
        key = digest(
            {
                "children": [
                    (n.node_id, n.content_sha256) for n in sorted(children, key=lambda n: n.node_id)
                ],
                "metadata": metadata,
            }
        )
        path = self.config.data / "cache" / "summaries" / f"{key}.json"
        cached = read_json(path)
        if cached:
            text = SummaryAnswer.model_validate(cached["summary"]).text
            self.progress(f"  Summary cache: nível {level}, {len(children)} children")
        else:
            answer = self.session.generate(
                "raptor_summary",
                self.summary_prompt,
                self.cluster_input(children),
                SummaryAnswer,
                self.config.summary_tokens,
                self.budget,
            )
            text = answer.text
            atomic_json(path, {"summary": answer.model_dump(), "metadata": metadata, "child_node_ids": ids})
        leaf_ids = sorted({leaf for child in children for leaf in child.source_leaf_ids})
        provenance = {canonical(p): p for child in children for p in child.provenance}
        return Node(
            node_id="s_" + digest({"key": key, "text": text, "level": level})[:20],
            node_type="summary",
            level=level,
            text=text,
            embedding_text=text,
            content_sha256=digest(text),
            child_node_ids=ids,
            source_leaf_ids=leaf_ids,
            doc_ids=sorted({d for n in children for d in n.doc_ids}),
            source_paths=sorted({p for n in children for p in n.source_paths}),
            provenance=list(provenance.values()),
            language=children[0].language,
            summary_metadata=metadata,
            embedding_fingerprint=self.embeddings.fingerprint,
        )

    def build(self, leaves):
        current = []
        for leaf in leaves:
            node = Node(
                node_id=leaf.leaf_id,
                node_type="leaf",
                level=0,
                text=leaf.text,
                embedding_text=leaf.embedding_text,
                content_sha256=leaf.content_sha256,
                source_leaf_ids=[leaf.leaf_id],
                doc_ids=[leaf.doc_id],
                source_paths=[leaf.source_path],
                provenance=[
                    {
                        "leaf_id": leaf.leaf_id,
                        "source_path": leaf.source_path,
                        "pdf_pages": leaf.pdf_pages,
                        "printed_page": leaf.printed_page,
                        "location": leaf.location,
                        "docling": leaf.provenance,
                    }
                ],
                language=leaf.language,
                embedding_fingerprint=self.embeddings.fingerprint,
            )
            self.nodes[node.node_id] = node
            current.append(node)
        self.progress(f"RAPTOR: {len(current)} leaves, embeddings sequenciais")
        for i, node in enumerate(current, 1):
            self.vectors[node.node_id] = self.embeddings.embed_documents([node.embedding_text])[0]
            if i == 1 or i % 25 == 0 or i == len(current):
                self.progress(
                    f"  Embeddings leaves: {i}/{len(current)}; cache reutilizado: "
                    f"{getattr(self.embeddings, 'cache_hits', 0)}"
                )
        stop_reason = "single_leaf_terminal" if len(current) == 1 else "max_levels"
        seen = set()
        self.checkpoint(0)
        for level in range(1, self.config.max_levels + 1):
            if len(current) <= 1:
                stop_reason = "single_node"
                break
            vectors = [self.vectors[n.node_id] for n in current]
            plan_path = self.build_dir / f"clusters_level_{level}.json"
            plan_key = digest(
                {
                    "nodes": [n.node_id for n in current],
                    "profile": self.config.profile(),
                    "embedding": self.embeddings.fingerprint,
                    "algorithm": "umap_gmm_bic_soft_v1",
                }
            )
            shared_plan = self.config.data / "cache" / "clusters" / f"{plan_key}.json"
            cached_plan = read_json(shared_plan) or read_json(plan_path)
            if not cached_plan:
                for candidate in sorted((self.config.data / "builds").glob(f"*/clusters_level_{level}.json")):
                    previous_plan = read_json(candidate)
                    if previous_plan and previous_plan.get("key") == plan_key:
                        cached_plan = previous_plan
                        break
            if cached_plan and cached_plan.get("key") == plan_key:
                groups = [[self.nodes[node_id] for node_id in ids] for ids in cached_plan["groups"]]
                self.cluster_reports = cached_plan.get("cluster_reports", [])
                atomic_json(plan_path, cached_plan)
                atomic_json(shared_plan, cached_plan)
                self.progress(f"  Clusters cache: nível {level}")
            else:
                groups = self.gmm_groups(current, vectors)
                groups = unique_groups([sub for group in groups for sub in self.subdivide(group)])
                atomic_json(
                    plan_path,
                    {
                        "key": plan_key,
                        "groups": [[n.node_id for n in g] for g in groups],
                        "cluster_reports": self.cluster_reports,
                    },
                )
                atomic_json(shared_plan, read_json(plan_path))
            signature = digest(sorted([sorted(n.node_id for n in group) for group in groups]))
            if signature in seen:
                stop_reason = "repeated_clusters"
                break
            seen.add(signature)
            if len(groups) >= len(current):
                if level == 1:
                    raise ValueError(
                        "Primeiro nível sem compressão; nenhum índice RAPTOR incompleto será ativado"
                    )
                stop_reason = "no_compression"
                break
            self.progress(f"RAPTOR nível {level}: {len(groups)} clusters para {len(current)} nós")
            next_nodes = []
            for i, group in enumerate(groups, 1):
                self.progress(f"  Summary {i}/{len(groups)}: {len(group)} children")
                parent = self.summary(group, level)
                self.nodes[parent.node_id] = parent
                next_nodes.append(parent)
                for child in group:
                    if parent.node_id not in child.parent_node_ids:
                        child.parent_node_ids.append(parent.node_id)
                self.checkpoint(level, cluster=i)
            # Batch phase: generation is finished before loading the embedding model.
            for node in next_nodes:
                self.vectors[node.node_id] = self.embeddings.embed_documents([node.embedding_text])[0]
            current = next_nodes
            self.checkpoint(level)
        self.checkpoint(max(n.level for n in self.nodes.values()), stop_reason=stop_reason)
        return list(self.nodes.values()), self.vectors, stop_reason

    def checkpoint(self, level, cluster=None, stop_reason=None):
        write_jsonl(self.build_dir / "tree.checkpoint.jsonl", self.nodes.values())
        atomic_json(
            self.build_dir / "checkpoint.json",
            {
                "level": level,
                "cluster": cluster,
                "nodes": len(self.nodes),
                "vectors": len(self.vectors),
                "status": "building",
                "stop_reason": stop_reason,
                "cluster_reports": self.cluster_reports,
            },
        )


def unique_groups(groups):
    unique = {}
    for group in groups:
        if group:
            ordered = sorted(group, key=lambda n: n.node_id)
            unique[tuple(n.node_id for n in ordered)] = ordered
    return [unique[key] for key in sorted(unique)]
