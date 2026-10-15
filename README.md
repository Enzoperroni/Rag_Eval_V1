# Especialista local xUnit

Sistema Python com **Query Decomposition de três etapas**, **RAPTOR** e **ChromaDB persistente**.
Toda geração usa `qwen3:8b`; todo embedding usa `batiai/qwen3-embedding:8b`, via Ollama local.

Os documentos originais ficam exclusivamente em `docs/`. O aplicativo não altera suas fontes.
Aceita PDF, TXT UTF-8, Markdown e DOCX, inclusive em subpastas e com extensões maiúsculas.
Arquivos não suportados aparecem no relatório. Falhas de conversão impedem a promoção do índice.

## Executar no PowerShell

Na raiz deste projeto, sem precisar ativar a `.venv`:

```powershell
.\.venv\Scripts\python.exe -m xunit_rag doctor
.\.venv\Scripts\python.exe -m xunit_rag prepare
.\.venv\Scripts\python.exe -m xunit_rag validate-index
.\.venv\Scripts\python.exe -m xunit_rag ask "Como corrigir testes que passam isolados e falham em suíte por estado compartilhado?"
.\.venv\Scripts\python.exe -m xunit_rag chat
.\.venv\Scripts\python.exe -m xunit_rag inspect --run-id ID
.\.venv\Scripts\python.exe -m xunit_rag evaluate --cases .\eval\cases.json
```

`chat` aceita `sair`, `exit` e Ctrl+C. O diretório das fontes é resolvido a partir do pacote instalado,
independentemente do diretório atual do terminal. Fora da raiz, chame o Python pelo caminho absoluto.
`inspect` exibe o trace JSONL da execução já existente, sem inferência.

Adicione/edite/remova suas fontes em `docs` e execute `prepare` novamente. O inventário usa caminho
relativo e SHA-256. Documentos e caches compatíveis são reutilizados. Documentos removidos deixam
de participar do novo índice. Abrir o chat não reconstrói o índice; se as fontes mudaram, a consulta
é interrompida com a indicação de executar `prepare`.

## Ambiente Python

A `.venv` Linux do pacote inicial foi substituída por uma `.venv` Windows Python 3.12.
As versões fixadas de NumPy exigem Python 3.12; esse é o runtime suportado neste projeto.
O runtime oficial usado na preparação está em `.runtime/python312`, restrito a este projeto.
Não mova o projeto com uma `.venv` já criada: recrie-a no destino.

Com Python 3.12 instalado:

```powershell
py -3.12 -m venv --clear .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
```

Se apenas o gerenciador Python estiver instalado, é possível extrair um runtime local:

```powershell
py install --target=.runtime\python312 3.12
.\.runtime\python312\python.exe -m venv --clear .venv
```

Os comandos `--clear` acima se destinam somente à `.venv` deste projeto. Não instale dependências
globalmente. Não é necessário alterar a ExecutionPolicy. `requirements.lock` contém as versões
efetivamente resolvidas e verificadas; inclui as dependências de desenvolvimento para reproduzir os checks.

## Ollama e memória

`doctor` consulta `/api/tags`, `/api/show`, capacidades, digests e faz um probe real da dimensão.
Os modelos devem estar instalados previamente. O aplicativo nunca executa `ollama pull`.
O servidor deve estar em execução; abra o Ollama normalmente ou execute `ollama serve` em um terminal.

Ao iniciar **um novo servidor**, defina nesse terminal:

```powershell
$env:OLLAMA_NUM_PARALLEL = "1"
$env:OLLAMA_MAX_LOADED_MODELS = "1"
ollama serve
```

Isso não altera um servidor que já está em execução. O ambiente interno de um servidor existente
não é exposto pela API e não é anunciado como verificado. A aplicação usa lock de inferência,
chamadas sequenciais, `/api/ps` e `keep_alive=0` para descarregar o outro modelo antes de alternar.
Não interrompe modelos alheios: se outro modelo estiver residente, pede que ele seja liberado.
Docling, UMAP e GMM usam CPU. Objetos da conversão são liberados antes das fases de inferência.

O perfil inicial usa `num_ctx=4096` para geração, `2048` para embeddings, lote de um texto,
um worker, timeout de 180 s por chamada, até três tentativas para erros transitórios e deadline de
1800 s por consulta. OOM interrompe sem troca de modelo. Offload observado aparece nos traces;
o arquivo dos pesos não prova que o modelo inteiro cabe na GPU. Não há promessa de velocidade.

Para mudar apenas o endereço local, defina `OLLAMA_HOST` no terminal. `.env.example` documenta as
variáveis; não é carregado automaticamente. As tags obrigatórias são validadas pela aplicação.

## Indexação

Docling produz `DoclingDocument`, Markdown e leaves com `HybridChunker.contextualize`.
O tokenizer oficial Qwen é baixado como JSON (sem pesos de geração/embedding Python), fixado por
revision/hash e usado para validar os **512 tokens do texto contextualizado inteiro**.
O HybridChunker usa margem interna de 32 tokens; a checagem final continua em 512, sem truncamento.
A conversão estruturada tem checkpoint próprio e pode ser reutilizada se o chunking precisar ser retomado.
No Windows, o backend PDF do Docling é explicitamente `PyPdfiumDocumentBackend`, evitando um
problema observado no acesso aos recursos nativos de `docling-parse` em caminhos Unicode.
Texto nativo é priorizado. Uma etapa Docling de OCR seletivo processa somente as páginas que a
inspeção identificou sem texto suficiente; as demais preservam suas células nativas. Os modelos de layout/tabelas/OCR do Docling
podem ser baixados na primeira conversão; ficam no cache do projeto e executam em CPU.

RAPTOR preserva os leaves, reduz os embeddings com UMAP quando N permite, seleciona GMM por BIC
e usa associações soft (threshold 0,1). Clusters idênticos são deduplicados; grupos grandes são
subdivididos semanticamente com limites e verificação de progresso. Não há truncamento de children.
Os summaries são gerados em uma fase, embedados em outra, e passam à próxima rodada, até cinco
níveis, um nó restante ou falta de compressão. Um leaf isolado é caso terminal sem summaries;
vários leaves sem summaries não constituem build completo. Falhas numéricas/orçamento preservam
checkpoint e impedem a ativação.

O graph permite múltiplos parents, sem ciclos. Chroma recebe vetores explícitos normalizados e
usa cosine, com `embedding_function=None`. A busca considera leaves e summaries de todos os níveis.
No Windows, um caminho Unicode usa um alias ASCII verificado da mesma pasta `data/chroma`,
necessário para os arquivos nativos HNSW nesta instalação. O banco não é movido nem copiado.
Execute os comandos em um PowerShell normal; um sandbox que bloqueia a consulta dos diretórios
pais pode impedir a resolução desse alias. Ausência de alias compatível produz erro explícito.

## Consulta e fontes

Cada pergunta usa uma chamada de decomposição, três recuperações e três respostas estritamente
sequenciais, seguidas de uma síntese: cinco chamadas lógicas LLM. Um único reparo de JSON do plano
pode elevar esse número a seis. Tentativas HTTP são contadas separadamente. Cada embedding de
consulta recebe somente a subpergunta validada e a instrução Qwen de retrieval.

R2 recebe os fatos de R1; R3 recebe os fatos de R1/R2. Hipóteses permanecem explicitamente
não verificadas. A síntese recebe os três resultados e as evidências registradas, sem quarta busca.
O orçamento inclui regras, pergunta, fontes, resultados anteriores, separadores, saída reservada e
margem. Passagens inteiras que não cabem são omitidas com ID e motivo no trace, sem cortar texto
silenciosamente. Pergunta/código grande demais produz erro explícito.
As passagens da busca atual recebem espaço antes das passagens completas anteriores; os fatos
anteriores e seus IDs continuam no contexto. Na síntese, as fontes dos fatos de R3 recebem espaço
primeiro, seguidas das demais evidências já enviadas, sem uma nova busca.

Citações `[l_...]` apontam para passages originais; `[s_...]` identificam summaries gerados,
com leaves de origem rastreáveis no graph. Páginas PDF só existem quando extraídas da proveniência.
Passagens identificadas como sumário/índice recebem um aviso de fonte de navegação no contexto;
continuam elegíveis na mesma busca cosine, mas não sustentam afirmações sobre aplicação de patterns.
Não se calcula offset de página impressa. Um summary não recebe uma página literal única e a união
das páginas dos descendants não comprova cada afirmação gerada.

As regras de especialista em `src/xunit_rag/prompts/decision_rules.md` não entram no índice.
IDs e estrutura válidos não provam fidelidade semântica: isso exige revisão do conteúdo das fontes.
As listas redundantes de fontes são calculadas dos fatos e citações já validados. A grafia
`[node_id:ID]` é padronizada como `[ID]`; IDs desconhecidos continuam sendo erros. Se o modelo indicar
referências gerais sem marcação inline, elas aparecem numa linha de fontes, com essa situação
registrada no trace; isso não atribui uma fonte a cada afirmação. Esses ajustes não alteram o conteúdo conceitual nem chamam o LLM;
a saída original e cada ajuste ficam registrados no trace.
Nenhum LLM-as-a-judge, busca híbrida, reranker, indexador alternativo ou modo de consulta direta é usado.
Respostas priorizam diagnóstico condicionado às evidências e ações curtas. Exemplos Java/JUnit 5
entram quando a pergunta pede código ou exemplo; sua compilação não é anunciada como verificada.

## Artefatos, retomada e avaliação

- `data/processed/ingest_corpus.jsonl`: corpus canônico, proveniência e política de tokens.
- `data/processed/documents/`: representações estruturadas, Markdown e cache por documento.
- `data/manifests/`: inventário, relatório por arquivo, doctor e ponteiro atômico do índice ativo.
- `data/builds/BUILD_ID/tree.jsonl`: graph completo, manifest e indicador `COMPLETE` do build.
- `data/builds/BUILD_ID/*checkpoint*`: progresso por nível/cluster.
- `data/chroma/`: banco persistente; coleções históricas ficam identificadas pelo build.
- `data/cache/`: embeddings, summaries e artefatos de conversão compatíveis.
- `logs/RUN_ID.jsonl`: plano, retrieval, contextos efetivos, evidências omitidas, tempos, chamadas e erros.
- `eval/`: resultados de avaliação, separados do custo de indexação.

`prepare` mantém lock exclusivo. A retomada reconstrói a estrutura deterministicamente e reutiliza
summaries/embeddings já concluídos, sem repetir sua inferência. Uma coleção staging incompleta
nunca substitui o ponteiro ativo. Não se renomeia banco aberto no Windows.
Ctrl+C preserva caches/checkpoints; consultas interrompidas recebem status próprio.
Trocas atômicas de arquivo têm tentativas limitadas para bloqueios transitórios do Windows;
se o bloqueio persistir, o destino anterior é preservado e a preparação falha explicitamente.

Exemplo do formato de casos (salve fora de `docs`, em `eval/cases.json`):

```json
[{"id":"shared-state","question":"Como evitar contaminação de estado compartilhado entre testes?"}]
```

`evaluate` usa exatamente o fluxo de `ask`. Registra falhas como falhas e deixa campos independentes
de revisão humana para decomposição, recuperação, fidelidade conceitual e utilidade do código.
Não calcula qualidade por contagem de palavras, não anuncia melhora sem comparação e não afirma
que exemplos Java foram compilados se isso não foi feito.

Checks locais:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp "data\tests_$([guid]::NewGuid().ToString('N'))"
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m pip check
```

Os testes usam corpus temporário e fakes somente em `tests`, separados de `docs` e do índice real.
A pasta exclusiva de cada execução evita reutilizar diretórios temporários criados por outra conta
ou sandbox. O teste de persistência abre o HNSW em um segundo processo Python.
Veja `project_notes/api_sources.md` para APIs oficiais consultadas e `project_notes/verification.md`
para os resultados efetivamente obtidos nesta máquina.
