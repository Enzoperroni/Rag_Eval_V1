"""Exact Qwen tokenizers; tokenizer JSON only, never Transformers model weights."""

import httpx
from tokenizers import Tokenizer

from .utils import atomic_json, digest, file_sha, read_json

REPOS = {"embedding": "Qwen/Qwen3-Embedding-8B", "llm": "Qwen/Qwen3-8B"}


class QwenTokenizer:
    def __init__(self, config, role="embedding"):
        self.role = role
        directory = config.data / "tokenizers" / role
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "tokenizer.json"
        meta_path = directory / "manifest.json"
        meta = read_json(meta_path)
        if not path.exists() or not meta:
            repo = REPOS[role]
            with httpx.Client(timeout=180, follow_redirects=True) as client:
                response = client.get(f"https://huggingface.co/api/models/{repo}")
                response.raise_for_status()
                revision = response.json()["sha"]
                response = client.get(f"https://huggingface.co/{repo}/resolve/{revision}/tokenizer.json")
                response.raise_for_status()
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(response.content)
                temporary.replace(path)
            meta = {"repo": repo, "revision": revision, "sha256": file_sha(path)}
            atomic_json(meta_path, meta)
        if file_sha(path) != meta["sha256"]:
            raise ValueError(f"Tokenizer corrompido: {path}")
        self.tokenizer = Tokenizer.from_file(str(path))
        self.policy = {**meta, "add_special_tokens": False, "method": "exact_tokenizer_json", "role": role}

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def get_max_tokens(self):
        return 512

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)


class PromptBudget:
    def __init__(self, config, tokenizer):
        self.config, self.tokenizer = config, tokenizer

    def count(self, system, user):
        # Qwen chat framing + disabled-thinking assistant prefix, with extra conservative margin below.
        rendered = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user} /no_think<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return self.tokenizer.count_tokens(rendered)

    def fits(self, system, user, output):
        return self.count(system, user) + output + self.config.margin_tokens <= self.config.num_ctx

    def require(self, system, user, output):
        count = self.count(system, user)
        if not self.fits(system, user, output):
            raise ValueError(
                f"Prompt excede num_ctx: input={count}, saída={output}, margem={self.config.margin_tokens}"
            )
        return {
            "input_tokens": count,
            "output_reserved": output,
            "margin": self.config.margin_tokens,
            "method": "Qwen tokenizer + conservative chat framing",
        }

    @property
    def fingerprint(self):
        return digest(self.tokenizer.policy)
