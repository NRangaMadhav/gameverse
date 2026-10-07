import base64
import hashlib
import logging
import mimetypes
import os
from pathlib import Path

import pymupdf
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA

from app.settings import (
    CHROMA_DIR,
    DEFAULT_PDF_DIR,
    nvidia_request_timeout,
    required_setting,
)

logger = logging.getLogger(__name__)
COLLECTION_NAME = "gameverse_india"


def _embeddings() -> NVIDIAEmbeddings:
    required_setting("NVIDIA_API_KEY")
    return NVIDIAEmbeddings(
        model=os.getenv("NVIDIA_EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5"),
    )


def _vector_store() -> Chroma:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=_embeddings(),
    )


def _figure_description(page: pymupdf.Page, image_xref: int, source: str) -> str:
    extracted = page.parent.extract_image(image_xref)
    image_bytes = extracted.get("image")
    if not image_bytes:
        raise ValueError(f"Could not extract image {image_xref} from {source}.")

    image_ext = extracted.get("ext", "png")
    mime_type = mimetypes.guess_type(f"figure.{image_ext}")[0] or "image/png"
    encoded_image = base64.b64encode(image_bytes).decode("ascii")
    required_setting("NVIDIA_API_KEY")
    model = ChatNVIDIA(
        model=os.getenv("NVIDIA_LLM_MODEL", "google/diffusiongemma-26b-a4b-it"),
        timeout=nvidia_request_timeout(),
    )
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "Describe the visual evidence in this gaming-industry report figure. "
                    "Read visible labels and values where possible. Be concise and do not guess."
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{encoded_image}"},
            },
        ]
    )
    response = model.invoke([message])
    description = str(response.content).strip()
    if not description:
        raise RuntimeError(f"NVIDIA returned an empty description for an image in {source}.")
    return description


def _extract_pdf(path: Path) -> list[Document]:
    documents: list[Document] = []
    source = path.name
    with pymupdf.open(path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            metadata = {"source": source, "page": page_number}
            text = page.get_text("text").strip()
            if text:
                documents.append(
                    Document(page_content=text, metadata={**metadata, "kind": "text"})
                )

            tables = page.find_tables()
            for table_number, table in enumerate(tables.tables, start=1):
                rows = table.extract()
                table_text = "\n".join(
                    " | ".join(str(cell or "").strip() for cell in row) for row in rows
                ).strip()
                if table_text:
                    documents.append(
                        Document(
                            page_content=f"Table {table_number}:\n{table_text}",
                            metadata={**metadata, "kind": "table"},
                        )
                    )

            seen_xrefs: set[int] = set()
            for image in page.get_images(full=True):
                image_xref = image[0]
                if image_xref in seen_xrefs:
                    continue
                seen_xrefs.add(image_xref)
                description = _figure_description(page, image_xref, source)
                documents.append(
                    Document(
                        page_content=f"Figure description: {description}",
                        metadata={**metadata, "kind": "figure"},
                    )
                )
    return documents


def _document_id(document: Document) -> str:
    identity = (
        f"{document.metadata['source']}|{document.metadata['page']}|"
        f"{document.metadata['kind']}|{document.page_content}"
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def ingest_pdfs(directory: Path = DEFAULT_PDF_DIR) -> dict[str, int]:
    if not directory.is_dir():
        raise FileNotFoundError(f"PDF directory does not exist: {directory}")
    pdf_paths = sorted(directory.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files were found in {directory}.")

    extracted: list[Document] = []
    for path in pdf_paths:
        logger.info("Extracting %s", path.name)
        extracted.extend(_extract_pdf(path))
    if not extracted:
        raise ValueError("No searchable text, tables, or figures were extracted from the PDFs.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1100, chunk_overlap=160)
    chunks = splitter.split_documents(extracted)
    store = _vector_store()
    sources = sorted({path.name for path in pdf_paths})
    for source in sources:
        store.delete(where={"source": source})
    store.add_documents(chunks, ids=[_document_id(document) for document in chunks])
    return {"files": len(pdf_paths), "documents": len(extracted), "chunks": len(chunks)}


def retrieve(question: str, limit: int = 6) -> list[Document]:
    store = _vector_store()
    if not store.get(limit=1).get("ids"):
        return []
    return store.similarity_search(question, k=limit)
