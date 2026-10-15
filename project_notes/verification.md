# Verificação local — 07/10/2026

Sistema implementado e preparado no workspace
`C:\Users\Usuário\Downloads\xunit_rag_raptor_inicial\xunit_rag_raptor`.

## Ambiente e modelos

- `.venv` recriada para Windows, Python 3.12.10; `sys.prefix != sys.base_prefix`.
- Dependências restritas à `.venv`; 173 versões resolvidas em `requirements.lock`.
- NumPy fixado exige Python 3.12; o metadata do projeto reflete esse requisito.
- Ollama 0.33.2: tags, show, embed, ps e descarregamento verificados.
- Geração: `qwen3:8b`, Q4_K_M, digest
  `500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41`.
- Embedding: `batiai/qwen3-embedding:8b`, Q6_K, digest
  `b38ba28db75cbdc2551f8e1b058bd451c55a18f18193021d6a3e97486c3d4240`.
- Dimensão **4096**, medida pelo endpoint real `/api/embed`, sem redução dimensional do vetor armazenado.
- Nenhum `ollama pull`, troca de modelo ou download duplicado de pesos Qwen.
- Tokenizers oficiais baixados como JSON, fixados por revision e SHA-256; assets Docling em cache local.
- TLS usa truststore/certificados do Windows, sem desabilitar validação HTTPS.

## Corpus e índice completos

Um documento encontrado e ingerido, sem exclusões ou arquivos não suportados:
`[xUnit Test Patterns Refactoring Test Code (Addison-Wesley Signature Series (Fowler)) Kindle Edition by Gerard Meszaros - 2007].pdf`.
Original preservado em `docs`, com 6.348.530 bytes e **948 páginas PDF**.
SHA-256: `d5e078c36e79a3adb6232cb53312713bb1fcb6f3f0805ceee38e7acf4cb41fa6`.

Docling converteu todas as páginas. Foram produzidos **2.214 leaves**, idioma `en`.
O maior texto contextualizado medido tem **492 tokens**, dentro do teto de 512.
Páginas impressas não foram inferidas; a proveniência guarda as páginas PDF reais.

Build ativo: `e446010a6dfede63f0e8da9dc5d40e2d34c0135beaa1028372b5f9040e36365d`.

| Nível | Nós |
| --- | ---: |
| 0 — leaves | 2214 |
| 1 | 268 |
| 2 | 24 |
| 3 | 23 |
| 4 | 22 |
| 5 | 21 |

Total: **358 summaries, cinco níveis de summaries, 2.572 nós**; parada por `max_levels`.
O graph tem **79 nós com múltiplos parents**. Children, níveis, unions de source_leaf_ids,
conteúdo, proveniência, fingerprints e vetores foram validados.

Chroma persistido em `data/chroma`, cosine e vetores explícitos normalizados de 4096 dimensões,
com `embedding_function=None`. `validate-index` passou em um **novo processo Python**.
Uma nova execução de `prepare` retornou **reused**, mantendo o mesmo build.
Na atualização dos prompts, os 2.572 embeddings e os 358 summaries compatíveis foram reutilizados.
Coleções e checkpoints históricos foram preservados; um staging incompleto não substitui o ativo.

## Testes e consultas reais

- **51 testes passaram** na execução completa, em 30,22 s.
- Dois avisos de depreciação do próprio Chroma sobre configuração de embedding function;
  os testes usam vetores explícitos e nenhuma inferência de embedding padrão.
- Ruff, compileall e `pip check` passaram.
- O teste de HNSW persiste 1.100 vetores temporários e abre/consulta a coleção em outro processo.
- `chat` iniciou no terminal e encerrou com `sair`.

Comando da suíte final:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp data\native_tests_release_20261007
```

Rodada real final: [evaluation_b3c579e52ec74ab2977c8184c4b35503.json](../eval/evaluation_b3c579e52ec74ab2977c8184c4b35503.json), **5/5 execuções concluídas**.
Isso significa sucesso operacional; não é uma nota de correção semântica.

| Caso | Trace |
| --- | --- |
| Estado compartilhado | [08c5f3029a24456b92bb5bd016177107](../logs/08c5f3029a24456b92bb5bd016177107.jsonl) |
| Test Stub / Mock Object | [7796edd9c46847ab88f91b5880743469](../logs/7796edd9c46847ab88f91b5880743469.jsonl) |
| Notificação / Behavior Verification | [2cf11fb0b2dc44308693fad314a50b22](../logs/2cf11fb0b2dc44308693fad314a50b22.jsonl) |
| ShoppingCart / State Verification | [bbde60d448a045908901b368219c27e9](../logs/bbde60d448a045908901b368219c27e9.jsonl) |
| Pergunta fora do corpus | [f304c169c43849f48a51b855006834fc](../logs/f304c169c43849f48a51b855006834fc.jsonl) |

Para cada trace, uma checagem dos eventos confirmou **cinco chamadas lógicas LLM e três buscas Chroma**,
sem reparo do plano. Os contextos anteriores foram exatamente `[]`, `[1]`, `[1,2]`, `[1,2,3]`.
Os inputs de retrieval são as subperguntas; respostas inteiras não são embedadas.
As consultas desta rodada reutilizaram os embeddings em cache: cinco tentativas de inferência
de geração, sem três novas inferências de embedding. As três buscas ocorreram normalmente.
As observações de residência mostraram no máximo um modelo por vez.
O resumo extraído dos traces está em [verified_trace_summary.json](../eval/verified_trace_summary.json).

A pergunta astronômica recebeu insuficiência documental explícita e `evidence_ids=[]`.
As outras respostas distinguiram estado/interações e registraram lacunas dos casos concretos.
Os resultados são gerais: não demonstram uma correção executada no código de um usuário.

## Ajustes verificados e limites

- O backend nativo padrão do Docling falhou num recurso em caminho Unicode. Foi usado o backend
  PDFium suportado pelo Docling, mantendo sua extração estruturada e HybridChunker.
- OCR seletivo somente nas páginas PDF 1 e 5, identificadas com texto nativo insuficiente;
  layout/estrutura das 948 páginas processados em CPU.
- Um chunk de 513 tokens foi rejeitado. A margem interna do HybridChunker passou a 32 tokens,
  preservando o limite final validado de 512 e reutilizando a conversão estruturada.
- Um summary atingiu o limite de saída e interrompeu o build. O prompt passou a exigir um
  parágrafo curto; os summaries anteriores incompatíveis foram invalidados pelo hash do prompt.
- O HNSW não persistia seus arquivos nativos corretamente no caminho Unicode desta instalação.
  Usa-se um alias ASCII do Windows verificado com `samefile`, para a **mesma** pasta `data/chroma`.
  Não houve mudança de pasta nem cópia de banco. Os arquivos `.bin` existem e a reabertura passou.
- Bloqueio transitório na troca de checkpoint do Windows é tratado por tentativas curtas limitadas.
  Bloqueio persistente mantém o destino anterior e o temporário completo e produz erro.
- Respostas incompletas do Ollama são rejeitadas, sem reparos ocultos. O JSON original é registrado.
  Listas redundantes de IDs são calculadas dos fatos/citações validados; grafia de citações é
  padronizada. Referências gerais podem aparecer em uma linha de fontes, sem atribuição por afirmação.
  IDs desconhecidos são sempre erros. Esses ajustes não usam LLM nem alteram conclusões conceituais.
- A revisão das primeiras rodadas revelou conselhos genéricos e código incompleto. A síntese foi
  restringida; passagens de navegação são identificadas no contexto. Resultados anteriores,
  inclusive falhas, continuam registrados; não foram convertidos em sucessos retrospectivos.
- Fontes/IDs corretos não provam fidelidade semântica. Os campos independentes de revisão humana
  permanecem disponíveis; não houve LLM-as-a-judge, score lexical ou comparação de arquiteturas.
- Nenhum exemplo Java gerado foi compilado/executado. Pedidos explícitos de código devem ser revisados.
- O servidor existente não expõe seu ambiente. As opções recomendadas de paralelismo/residência
  foram confirmadas no help, mas não anunciadas como configuração efetiva do servidor.
- Probe de embedding observou contexto 2048 e `size_vram=6.286.328.462`; geração observou contexto
  4096 e `size_vram=5.578.204.118`. Esses números não demonstram offload integral para a GPU.
- Os comandos que resolvem o alias nativo foram executados fora do sandbox restritivo. Testes usam
  pastas temporárias exclusivas para evitar ACLs herdadas de outra conta/sandbox.

Comandos de instalação e uso: [README.md](../README.md). APIs primárias: [api_sources.md](api_sources.md).
