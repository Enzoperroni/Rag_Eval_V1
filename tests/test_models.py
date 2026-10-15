import json
import httpx
import numpy as np
import pytest
from xunit_rag.models import ModelError, OllamaSession, QwenEmbeddings, normalized_vector
from xunit_rag.schemas import SummaryAnswer


def session_with_server(config, handler):
    session = OllamaSession(config)
    session.client.close()
    session.client = httpx.Client(transport=httpx.MockTransport(handler), base_url=config.ollama_host)
    return session


def test_embedding_wrappers_cache_digest_and_dimension(config, tokenizer):
    inputs = []

    def handler(request):
        body = json.loads(request.content) if request.content else {}
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        if request.url.path == "/api/embed":
            assert body["truncate"] is False and body["options"]["num_ctx"] == 2048
            inputs.append(body["input"][0])
            return httpx.Response(200, json={"embeddings": [[3, 4, 0]]})
        return httpx.Response(200, json={})

    session = session_with_server(config, handler)
    info = {"models": {config.embedding_model: {"digest": "digest_a"}}}
    adapter = QwenEmbeddings(config, session, tokenizer, info)
    assert adapter.dimension == 3
    assert np.isclose(np.linalg.norm(adapter.embed_documents(["Original passage"])[0]), 1)
    adapter.embed_query("PaymentService Test Stub?")
    assert inputs[-1].startswith("Instruct: ") and "\nQuery: PaymentService" in inputs[-1]
    assert inputs[1] == "Original passage"
    assert all("search_document:" not in text and "search_query:" not in text for text in inputs)
    before = len(inputs)
    adapter.embed_documents(["Original passage"])
    assert len(inputs) == before
    assert adapter.cache_hits == 1 and adapter.cache_misses == 2
    different = QwenEmbeddings(
        config, session, tokenizer, {"models": {config.embedding_model: {"digest": "digest_b"}}}
    )
    assert different.dimension == adapter.dimension and different.fingerprint != adapter.fingerprint
    different.embed_documents(["Original passage"])
    assert len(inputs) == before + 2  # one probe and one uncached passage
    session.close()


@pytest.mark.parametrize("values", [[0, 0], [float("nan"), 1], [[1, 2]], []])
def test_bad_vectors(values):
    with pytest.raises(ModelError):
        normalized_vector(values)


def test_dimension_mismatch():
    with pytest.raises(ModelError, match="Dimensão"):
        normalized_vector([1, 2, 3], 2)


def test_residency_unloads_other_role_before_call(config):
    loaded, events = [config.llm_model], []

    def handler(request):
        body = json.loads(request.content) if request.content else {}
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": name} for name in loaded]})
        if body.get("keep_alive") == 0:
            events.append(("unload", body["model"]))
            loaded.clear()
        return httpx.Response(200, json={})

    session = session_with_server(config, handler)
    with session.using(config.embedding_model):
        events.append(("infer", config.embedding_model))
        assert loaded == []
    assert events[:2] == [("unload", config.llm_model), ("infer", config.embedding_model)]
    session.close()


def test_oom_has_no_retry_or_model_swap(config):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(500, json={"error": "CUDA out of memory"})

    session = session_with_server(config, handler)
    with pytest.raises(ModelError, match="OOM"):
        session.request("/api/embed", {"model": config.embedding_model}, inference=True)
    assert len(calls) == 1


def test_offline_retries_bounded(config, monkeypatch):
    calls = []

    def handler(request):
        calls.append(1)
        raise httpx.ConnectError("offline", request=request)

    monkeypatch.setattr("xunit_rag.models.time.sleep", lambda _: None)
    session = session_with_server(config, handler)
    with pytest.raises(ModelError, match="3 tentativas"):
        session.request("/api/embed", {}, inference=True)
    assert len(calls) == 3 and session.transport_attempts == 3


def test_deadline_stops_before_request(config):
    session = OllamaSession(config, deadline=0.001)
    with pytest.raises(TimeoutError):
        session.request("/api/tags")
    session.close()


@pytest.mark.parametrize("done,reason", [(False, None), (True, "length")])
def test_unfinished_generation_stops_without_hidden_repair(config, tokenizer, done, reason):
    from xunit_rag.tokenization import PromptBudget

    inference_calls = []

    def handler(request):
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        if request.url.path == "/api/chat":
            inference_calls.append(1)
            return httpx.Response(200, json={
                "message": {"content": '{"text":"unfinished'}, "done": done, "done_reason": reason
            })
        return httpx.Response(200, json={})

    session = session_with_server(config, handler)
    with pytest.raises(ModelError, match="sem reparo oculto"):
        session.generate("summary", "Summarize", "Source", SummaryAnswer, 256,
                         PromptBudget(config, tokenizer))
    assert len(inference_calls) == session.logical_calls == session.transport_attempts == 1
    session.close()
