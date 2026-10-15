import threading
import time
from contextlib import contextmanager

import httpx
import numpy as np
from filelock import FileLock
from langchain_core.embeddings import Embeddings

from .config import QUERY_INSTRUCTION
from .utils import atomic_json, digest, read_json


class ModelError(RuntimeError):
    pass


class OllamaSession:
    def __init__(self, config, trace=None, deadline=None):
        self.config, self.trace, self.deadline = config, trace, deadline
        self.client = httpx.Client(base_url=config.ollama_host, timeout=config.call_timeout, trust_env=False)
        self.resident = None
        self.logical_calls = 0
        self.transport_attempts = 0
        self.lock = threading.RLock()
        self.process_lock = FileLock(str(config.data / "inference.lock"), timeout=0)

    def emit(self, event, **fields):
        if self.trace:
            self.trace.emit(event, **fields)

    def remaining(self):
        remaining = self.deadline - time.monotonic() if self.deadline else self.config.call_timeout
        if remaining <= 0:
            raise TimeoutError("Deadline de consulta excedido")
        return min(self.config.call_timeout, remaining)

    def request(self, endpoint, payload=None, inference=False, cleanup=False):
        for attempt in range(1, self.config.max_attempts + 1):
            start = time.monotonic()
            try:
                timeout = min(15, self.config.call_timeout) if cleanup else self.remaining()
                if inference:
                    self.transport_attempts += 1
                response = self.client.request(
                    "GET" if payload is None else "POST", endpoint, json=payload, timeout=timeout
                )
                if response.status_code >= 400:
                    message = response.text
                    if any(word in message.lower() for word in ("out of memory", "cuda error", "oom")):
                        raise ModelError(f"Ollama OOM; modelos preservados: {message}")
                    if response.status_code not in (429, 502, 503, 504):
                        raise ModelError(f"Ollama {response.status_code}: {message}")
                    response.raise_for_status()
                result = response.json()
                if "error" in result:
                    raise ModelError(str(result["error"]))
                self.emit(
                    "transport",
                    endpoint=endpoint,
                    attempt=attempt,
                    seconds=time.monotonic() - start,
                    inference=inference,
                    load_duration_ns=result.get("load_duration"),
                    total_duration_ns=result.get("total_duration"),
                    prompt_eval_count=result.get("prompt_eval_count"),
                    eval_count=result.get("eval_count"),
                )
                return result
            except (httpx.TransportError, httpx.HTTPStatusError) as error:
                self.emit(
                    "transport_error",
                    endpoint=endpoint,
                    attempt=attempt,
                    error=str(error),
                    seconds=time.monotonic() - start,
                )
                if attempt == self.config.max_attempts:
                    raise ModelError(f"Ollama indisponível após {attempt} tentativas: {error}") from error
                time.sleep(min(2 ** (attempt - 1), 2 if cleanup else self.remaining()))
        raise AssertionError("Tentativas esgotadas")

    def inspect_models(self):
        version = self.request("/api/version")
        tags = self.request("/api/tags")["models"]
        by_name = {m["name"]: m for m in tags}
        info = {"ollama_version": version["version"], "models": {}}
        for tag, capability in (
            (self.config.llm_model, "completion"),
            (self.config.embedding_model, "embedding"),
        ):
            if tag not in by_name:
                raise ModelError(
                    f"Modelo ausente: {tag}. Observados: {sorted(by_name)}. Nenhum peso será baixado."
                )
            show = self.request("/api/show", {"model": tag})
            if capability not in show.get("capabilities", []):
                raise ModelError(f"{tag} não declara capacidade {capability}")
            template = show.get("template", "")
            if tag == self.config.embedding_model and ("Instruct:" in template or "Query:" in template):
                raise ModelError(
                    "Template embedding já contém instrução; verificar para evitar wrapper duplo"
                )
            info["models"][tag] = {
                "digest": by_name[tag]["digest"],
                "details": show.get("details", {}),
                "model_info": show.get("model_info", {}),
                "capabilities": show["capabilities"],
                "template_sha256": digest(template),
                "parameters": show.get("parameters", ""),
            }
        self.model_info = info
        return info

    def unload(self, tag):
        self.request("/api/generate", {"model": tag, "keep_alive": 0}, cleanup=True)
        self.emit("model_unloaded", model=tag)
        if self.resident == tag:
            self.resident = None

    @contextmanager
    def using(self, tag):
        with self.lock, self.process_lock:
            loaded = self.request("/api/ps").get("models", [])
            for model in loaded:
                name = model.get("name", model.get("model"))
                if name != tag:
                    if name not in (self.config.llm_model, self.config.embedding_model):
                        raise ModelError(
                            f"Modelo externo residente: {name}; libere-o no Ollama antes de continuar"
                        )
                    self.unload(name)
            self.resident = tag
            try:
                yield
                observed = self.request("/api/ps").get("models", [])
                self.emit("residency", model=tag, observed=observed)
            except BaseException:
                try:
                    self.unload(tag)
                except Exception as unload_error:
                    self.emit("unload_error", error=str(unload_error))
                raise

    def generate(self, role, system, user, schema, output_tokens, budget):
        tokens = budget.require(system, user, output_tokens)
        self.logical_calls += 1
        self.emit("llm_call", role=role, logical_call=self.logical_calls, budget=tokens)
        with self.using(self.config.llm_model):
            result = self.request(
                "/api/chat",
                {
                    "model": self.config.llm_model,
                    "stream": False,
                    "think": False,
                    "keep_alive": "5m",
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    "format": schema.model_json_schema(),
                    "options": {
                        "num_ctx": self.config.num_ctx,
                        "num_predict": output_tokens,
                        "temperature": self.config.temperature,
                        "seed": self.config.seed,
                    },
                },
                inference=True,
            )
        text = result["message"]["content"]
        self.emit(
            "llm_result",
            role=role,
            content=text,
            prompt_eval_count=result.get("prompt_eval_count"),
            eval_count=result.get("eval_count"),
            done=result.get("done"),
            done_reason=result.get("done_reason"),
        )
        if result.get("done") is not True:
            raise ModelError(f"Ollama retornou resposta {role} não concluída; sem reparo oculto")
        if result.get("done_reason") == "length":
            raise ModelError(f"Saída {role} atingiu limite de {output_tokens} tokens; sem reparo oculto")
        return schema.model_validate_json(text)

    def close(self):
        try:
            if self.resident:
                with self.lock, self.process_lock:
                    self.unload(self.resident)
        finally:
            self.client.close()


def normalized_vector(values, dimension=None):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ModelError("Embedding inválido: forma/valores não finitos")
    if dimension is not None and len(array) != dimension:
        raise ModelError(f"Dimensão inconsistente: {len(array)} != {dimension}")
    norm = np.linalg.norm(array)
    if norm <= 1e-12:
        raise ModelError("Embedding nulo")
    return (array / norm).tolist()


class QwenEmbeddings(Embeddings):
    def __init__(self, config, session, tokenizer, model_info, measured_dimension=None):
        self.config, self.session, self.tokenizer = config, session, tokenizer
        self.model = config.embedding_model
        self.dimension = measured_dimension
        self.cache_hits = 0
        self.cache_misses = 0
        self.policy = {
            "model": self.model,
            "digest": model_info["models"][self.model]["digest"],
            "instruction": QUERY_INSTRUCTION,
            "document_prefix": "",
            "normalization": "l2_float64",
            "tokenizer": tokenizer.policy,
            "num_ctx": config.embedding_ctx,
            "truncate": False,
        }
        # doctor/prepare measure the endpoint. Queries reuse that measured dimension from a compatible
        # complete index, and every actual vector is still checked. No extra inference before decomposition.
        if self.dimension is None:
            self.dimension = len(self._embed("Software testing embedding dimension probe.", cache=False))
            session.emit("embedding_probe", dimension=self.dimension, method="actual_api_probe")
        elif not isinstance(self.dimension, int) or self.dimension <= 0:
            raise ModelError("Dimensão medida do índice inválida")
        self.policy["dimension"] = self.dimension
        self.fingerprint = digest(self.policy)
        session.emit("embedding_policy", dimension=self.dimension, policy=self.policy)

    @staticmethod
    def query_input(text):
        return f"Instruct: {QUERY_INSTRUCTION}\nQuery: {text}"

    def cache_key(self, text):
        return digest({"text": text, "policy": self.policy})

    def _embed(self, text, cache=True):
        count = self.tokenizer.count_tokens(text)
        if count + 8 > self.config.embedding_ctx:
            raise ModelError(f"Embedding excede contexto ({count} tokens + especiais); truncate=false")
        key = self.cache_key(text)
        path = self.config.data / "cache" / "embeddings" / f"{key}.json"
        cached = read_json(path) if cache else None
        if cached:
            if cached.get("key") != key or cached.get("policy") != self.policy:
                raise ModelError("Cache de embedding incompatível/corrompido")
            vector = normalized_vector(cached["vector"], self.dimension)
            self.cache_hits += 1
            self.session.emit("embedding_cache", key=key, hit=True)
            return vector
        with self.session.using(self.model):
            result = self.session.request(
                "/api/embed",
                {
                    "model": self.model,
                    "input": [text],
                    "truncate": False,
                    "keep_alive": "5m",
                    "options": {"num_ctx": self.config.embedding_ctx},
                },
                inference=True,
            )
        vectors = result.get("embeddings", [])
        if len(vectors) != 1:
            raise ModelError("Quantidade inesperada de embeddings")
        vector = normalized_vector(vectors[0], self.dimension)
        if cache:
            atomic_json(path, {"key": key, "policy": self.policy, "vector": vector})
            self.cache_misses += 1
        self.session.emit(
            "embedding",
            input=text,
            token_count=count,
            measured_prompt_tokens=result.get("prompt_eval_count"),
            key=key,
            dimension=len(vector),
        )
        return vector

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(self.query_input(text))
