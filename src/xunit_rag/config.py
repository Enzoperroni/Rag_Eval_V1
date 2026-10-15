import os
from dataclasses import asdict, dataclass
from pathlib import Path

LLM_MODEL = "qwen3:8b"
EMBEDDING_MODEL = "batiai/qwen3-embedding:8b"
QUERY_INSTRUCTION = (
    "Retrieve relevant passages about software testing and xUnit Test Patterns that answer the query."
)
SCHEMA_VERSION = "1"
PROMPT_VERSION = "1"


@dataclass(frozen=True)
class Config:
    root: Path
    llm_model: str = LLM_MODEL
    embedding_model: str = EMBEDDING_MODEL
    ollama_host: str = "http://localhost:11434"
    num_ctx: int = 4096
    embedding_ctx: int = 2048
    chunk_tokens: int = 512
    top_k: int = 5
    seed: int = 224
    reduction_dim: int = 10
    threshold: float = 0.1
    max_clusters: int = 50
    max_levels: int = 5
    summary_tokens: int = 256
    cluster_tokens: int = 2800
    temperature: float = 0.1
    call_timeout: int = 180
    query_deadline: int = 1800
    max_attempts: int = 3
    margin_tokens: int = 160

    @classmethod
    def load(cls):
        root = Path(__file__).resolve().parents[2]
        for name, required in (("LLM_MODEL", LLM_MODEL), ("EMBEDDING_MODEL", EMBEDDING_MODEL)):
            if os.environ.get(name, required) != required:
                raise ValueError(f"{name} deve ser {required}")
        return cls(root=root, ollama_host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))

    @property
    def docs(self):
        return self.root / "docs"

    @property
    def data(self):
        return self.root / "data"

    def profile(self):
        return {k: v for k, v in asdict(self).items() if k not in {"root", "ollama_host"}}

    def initialize(self):
        # Trust the OS certificate store (including locally installed corporate roots); never disable TLS.
        import truststore

        truststore.inject_into_ssl()
        for name in ("processed", "chroma", "cache", "manifests", "builds", "tokenizers"):
            (self.data / name).mkdir(parents=True, exist_ok=True)
        for name in ("logs", "eval", "project_notes"):
            (self.root / name).mkdir(exist_ok=True)
        # Conversion models/cache stay inside this project; never in docs or the user's global cache.
        for name in ("HF_HOME", "TORCH_HOME", "DOCLING_ARTIFACTS_PATH"):
            if name != "DOCLING_ARTIFACTS_PATH":
                os.environ.setdefault(name, str(self.data / "cache" / name.lower()))
        os.environ["ANONYMIZED_TELEMETRY"] = "False"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["OTEL_SDK_DISABLED"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["DO_NOT_TRACK"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
