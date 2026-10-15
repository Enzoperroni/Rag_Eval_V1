# APIs conferidas

Fontes primárias consultadas em 07/10/2026, junto com imports e assinaturas da instalação:

- [Docling: chunking e contextualize](https://docling-project.github.io/docling/concepts/chunking/)
- [Docling BaseTokenizer](https://github.com/docling-project/docling-core/blob/main/docling_core/transforms/chunker/tokenizer/base.py)
- [Ollama /api/embed: truncate=false, keep_alive e options](https://docs.ollama.com/api/embed)
- [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs)
- [Ollama chat: resposta concluída e metadados](https://docs.ollama.com/api/chat)
- [Ollama thinking](https://docs.ollama.com/capabilities/thinking)
- [Ollama memória e opções do servidor](https://docs.ollama.com/faq)
- [Chroma persistência e vetores explícitos](https://docs.trychroma.com/docs/overview/getting-started)
- [RAPTOR, Sarthi et al.](https://arxiv.org/abs/2401.18059)
- [Qwen3-Embedding-8B e instruction wrapper](https://huggingface.co/Qwen/Qwen3-Embedding-8B)
- [Python venv](https://docs.python.org/3/library/venv.html)
- [Python install manager: target local](https://docs.python.org/3/using/windows.html)
- [Windows GetShortPathNameW: alias nativo e restrições de acesso](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-getshortpathnamew)

A aplicação usa somente componentes de embeddings de LangChain, um adaptador HTTP fino de Ollama,
e não invoca chains, agentes, tracing remoto ou modelos Python de geração/embedding.
