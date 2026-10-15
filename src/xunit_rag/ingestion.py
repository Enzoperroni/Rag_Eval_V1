import gc
import importlib.metadata
import re
import threading
import time

from langdetect import DetectorFactory, LangDetectException, detect

from .config import SCHEMA_VERSION
from .schemas import Leaf
from .utils import atomic_json, atomic_text, digest, file_sha, read_json, read_jsonl, utc_now, write_jsonl

SUPPORTED = {".pdf", ".txt", ".md", ".docx"}
DetectorFactory.seed = 224


class SelectiveOcr:
    """Keep native cells intact; run Docling OCR only on pages lacking native text."""

    def __init__(self, model, page_numbers):
        self.model, self.page_numbers = model, set(page_numbers)

    def __call__(self, conversion, pages):
        for page in pages:
            if page.page_no in self.page_numbers:
                yield from self.model(conversion, [page])
            else:
                yield page


class PageProgress:
    def __init__(self, model, total, progress):
        self.model, self.total, self.progress = model, total, progress
        self.seen = set()

    def __call__(self, conversion, pages):
        for page in self.model(conversion, pages):
            self.seen.add(page.page_no)
            count = len(self.seen)
            if count == 1 or count % 25 == 0 or count == self.total:
                self.progress(f"  Estrutura de páginas extraída: {count}/{self.total}")
            yield page


class DocumentIngestor:
    def __init__(self, config, tokenizer, progress=print):
        self.config, self.tokenizer, self.progress = config, tokenizer, progress
        self.manifest_path = config.data / "manifests" / "ingestion_manifest.json"
        self.policy = {
            "schema": SCHEMA_VERSION,
            "chunk_tokens": config.chunk_tokens,
            "tokenizer": tokenizer.policy,
            "docling": importlib.metadata.version("docling"),
            "docling_core": importlib.metadata.version("docling-core"),
            "parser_version": "3",
            "pdf_backend": "PyPdfiumDocumentBackend",
            "hybrid_max_tokens": max(1, config.chunk_tokens - 32),
        }
        self.conversion_policy = {
            key: self.policy[key] for key in ("docling", "docling_core", "parser_version", "pdf_backend")
        }

    def scan(self):
        return scan_docs(self.config)

    def changes(self, files=None):
        files = self.scan() if files is None else files
        old = read_json(self.manifest_path, {}).get("files", {})
        new = {f["path"]: f for f in files}
        return {
            "added": sorted(set(new) - set(old)),
            "removed": sorted(set(old) - set(new)),
            "changed": sorted(p for p in set(old) & set(new) if old[p]["sha256"] != new[p]["sha256"]),
        }

    def ingest(self):
        inventory = self.scan()
        changes = self.changes(inventory)
        all_leaves, reports, failures = [], {}, []
        self.progress(f"Inventário: {len(inventory)} arquivo(s); alterações: {changes}")
        for entry in inventory:
            path = self.config.docs / entry["path"]
            report = {**entry, "status": "unsupported" if not entry["supported"] else "pending"}
            if not entry["supported"]:
                self.progress(f"Não suportado: {entry['path']}")
                reports[entry["path"]] = report
                continue
            doc_id = "d_" + digest({"path": entry["path"], "sha256": entry["sha256"]})[:20]
            directory = self.config.data / "processed" / "documents" / doc_id
            cache = directory / "leaves.jsonl"
            meta = read_json(directory / "manifest.json", {})
            try:
                if cache.exists() and meta.get("policy") == self.policy and meta.get("status") == "complete":
                    leaves = [Leaf.model_validate(v) for v in read_jsonl(cache)]
                    report.update(status="reused", language=meta["language"], leaves=len(leaves))
                    self.progress(f"Cache ingestão: {entry['path']} ({len(leaves)} leaves)")
                else:
                    self.progress(f"Convertendo: {entry['path']}")
                    directory.mkdir(parents=True, exist_ok=True)
                    leaves, language, conversion = self.convert(path, entry, doc_id, directory)
                    write_jsonl(cache, leaves)
                    atomic_json(
                        directory / "manifest.json",
                        {
                            "status": "complete",
                            "policy": self.policy,
                            "language": language,
                            "conversion": conversion,
                        },
                    )
                    report.update(
                        status="ingested", language=language, leaves=len(leaves), conversion=conversion
                    )
                    self.progress(f"Ingerido: {entry['path']} ({len(leaves)} leaves, idioma={language})")
                if not leaves:
                    raise ValueError("Documento não produziu texto/leaves; verificar parser ou OCR")
                if file_sha(path) != entry["sha256"]:
                    raise ValueError("Fonte mudou durante a ingestão; execute prepare novamente")
                all_leaves.extend(leaves)
            except Exception as error:
                report.update(status="failed", error=f"{type(error).__name__}: {error}")
                failures.append(entry["path"])
                self.progress(f"Falha: {entry['path']}: {error}")
            reports[entry["path"]] = report
            atomic_json(self.config.data / "manifests" / "ingestion_report.json", reports)
        manifest = {
            "created_at": utc_now(),
            "policy": self.policy,
            "files": reports,
            "changes": changes,
            "corpus_hash": digest([leaf.model_dump() for leaf in all_leaves]),
            "inventory_hash": digest(inventory),
            "leaf_count": len(all_leaves),
            "status": "failed" if failures else "complete",
        }
        if failures:
            atomic_json(self.config.data / "manifests" / "ingestion_failed.json", manifest)
            raise RuntimeError(f"Ingestão incompleta: {failures}. Nenhum índice será promovido.")
        write_jsonl(self.config.data / "processed" / "ingest_corpus.jsonl", all_leaves)
        atomic_json(self.manifest_path, manifest)
        gc.collect()
        return all_leaves, manifest

    def convert(self, path, entry, doc_id, directory):
        from docling.chunking import HybridChunker
        from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
        from docling_core.types.doc import DocItemLabel, DoclingDocument

        class ChunkTokenizer(BaseTokenizer):
            def count_tokens(inner, text):
                return self.tokenizer.count_tokens(text)

            def get_max_tokens(inner):
                return self.policy["hybrid_max_tokens"]

            def get_tokenizer(inner):
                return self.tokenizer.tokenizer

        locations = {}
        conversion = {"format": path.suffix.lower(), "device": "cpu", "ocr": False}
        converted_path = directory / "document.json"
        converted = read_json(directory / "conversion_manifest.json", {})
        if (
            path.suffix.lower() != ".txt"
            and converted_path.exists()
            and converted.get("policy") == self.conversion_policy
            and converted.get("source_sha256") == entry["sha256"]
            and converted.get("document_sha256") == file_sha(converted_path)
        ):
            document = DoclingDocument.load_from_json(converted_path)
            conversion = converted["conversion"]
            self.progress(f"  Conversão estruturada reutilizada: {path.name}")
        elif path.suffix.lower() == ".txt":
            text = path.read_text(encoding="utf-8-sig", errors="strict")
            document = DoclingDocument(name=path.name)
            for match in re.finditer(r"\S[\s\S]*?(?=\n\s*\n|\Z)", text):
                item = document.add_text(label=DocItemLabel.TEXT, text=match.group())
                locations[item.self_ref] = {
                    "line_start": text.count("\n", 0, match.start()) + 1,
                    "line_end": text.count("\n", 0, match.end()) + 1,
                }
        else:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                AcceleratorDevice,
                AcceleratorOptions,
                PdfPipelineOptions,
            )
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
            from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

            options = PdfPipelineOptions()
            options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CPU, num_threads=4)
            needs_ocr, total_pages = [], 0
            if path.suffix.lower() == ".pdf":
                from pypdf import PdfReader

                reader = PdfReader(str(path))
                needs_ocr = [
                    i + 1
                    for i, page in enumerate(reader.pages)
                    if len((page.extract_text() or "").strip()) < 20
                ]
                options.do_ocr = bool(needs_ocr)
                total_pages = len(reader.pages)
                conversion.update(
                    ocr=bool(needs_ocr), ocr_candidate_pdf_pages=needs_ocr, pdf_page_count=len(reader.pages)
                )
                del reader
            else:
                options.do_ocr = False

            class SourcePdfPipeline(StandardPdfPipeline):
                def __init__(inner, pipeline_options):
                    super().__init__(pipeline_options)
                    inner.ocr_model = SelectiveOcr(inner.ocr_model, needs_ocr)
                    inner.assemble_model = PageProgress(inner.assemble_model, total_pages, self.progress)

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=options,
                        backend=PyPdfiumDocumentBackend,
                        pipeline_cls=SourcePdfPipeline,
                    )
                }
            )
            finished = threading.Event()
            started = time.monotonic()

            def report_conversion():
                while not finished.wait(30):
                    self.progress(f"  Conversão em andamento: {path.name}; {time.monotonic() - started:.0f}s")

            reporter = threading.Thread(target=report_conversion, daemon=True)
            reporter.start()
            try:
                result = converter.convert(path)
            finally:
                finished.set()
                reporter.join(timeout=1)
            if str(result.status.value) != "success":
                raise ValueError(f"Conversão incompleta: {result.status}; {result.errors}")
            document = result.document
            del result, converter
            gc.collect()
        document.save_as_json(directory / "document.json")
        atomic_text(directory / "document.md", document.export_to_markdown())
        atomic_json(
            directory / "conversion_manifest.json",
            {
                "policy": self.conversion_policy,
                "source_sha256": entry["sha256"],
                "conversion": conversion,
                "document_sha256": file_sha(directory / "document.json"),
            },
        )
        sample = " ".join(item.text for item, _ in document.iterate_items() if hasattr(item, "text"))[:30000]
        try:
            language = detect(sample) if sample.strip() else "unknown"
        except LangDetectException:
            language = "unknown"
        chunker = HybridChunker(tokenizer=ChunkTokenizer(), merge_peers=True)
        leaves = []
        for order, chunk in enumerate(chunker.chunk(dl_doc=document)):
            embedding_text = chunker.contextualize(chunk=chunk)
            count = self.tokenizer.count_tokens(embedding_text)
            if count > self.config.chunk_tokens:
                raise ValueError(
                    f"Chunk contextualizado {order} excede 512 tokens ({count}); sem truncamento"
                )
            refs, pages = [], set()
            for item in chunk.meta.doc_items:
                provenance = [p.model_dump(mode="json") for p in item.prov]
                pages.update(p.page_no for p in item.prov)
                refs.append(
                    {
                        "docling_ref": item.self_ref,
                        "label": str(item.label.value),
                        "provenance": provenance,
                        **locations.get(item.self_ref, {}),
                    }
                )
            headings = chunk.meta.headings or []
            content_hash = digest(chunk.text)
            leaves.append(
                Leaf(
                    leaf_id="l_"
                    + digest({"doc": doc_id, "order": order, "content": content_hash, "policy": self.policy})[
                        :20
                    ],
                    doc_id=doc_id,
                    source_path=entry["path"],
                    original_name=path.name,
                    source_sha256=entry["sha256"],
                    text=chunk.text,
                    embedding_text=embedding_text,
                    content_sha256=content_hash,
                    language=language,
                    headings=headings,
                    section=headings[-1] if headings else None,
                    provenance=refs,
                    pdf_pages=sorted(pages) if path.suffix.lower() == ".pdf" else [],
                    location={
                        "docling_refs": [r["docling_ref"] for r in refs],
                        "chunk_order": order,
                        "headings": headings,
                        "text_lines": [
                            locations[r["docling_ref"]] for r in refs if r["docling_ref"] in locations
                        ],
                    },
                    order=order,
                    token_policy={
                        **self.tokenizer.policy,
                        "contextualized_token_count": count,
                        "max_tokens": self.config.chunk_tokens,
                        "hybrid_max_tokens": self.policy["hybrid_max_tokens"],
                    },
                )
            )
        del document, chunker
        gc.collect()
        return leaves, language, conversion


def scan_docs(config):
    if not config.docs.is_dir():
        raise FileNotFoundError(f"Pasta de fontes ausente: {config.docs}")
    files = []
    for path in sorted(
        config.docs.rglob("*"), key=lambda p: p.relative_to(config.docs).as_posix().casefold()
    ):
        if path.is_file():
            if path.is_symlink() or not path.resolve().is_relative_to(config.docs.resolve()):
                raise ValueError(f"Fonte fora de docs por link: {path}")
            files.append(
                {
                    "path": path.relative_to(config.docs).as_posix(),
                    "sha256": file_sha(path),
                    "size": path.stat().st_size,
                    "supported": path.suffix.lower() in SUPPORTED,
                }
            )
    return files
