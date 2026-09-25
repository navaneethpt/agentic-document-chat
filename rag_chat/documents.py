"""Extract text without persisting uploaded files, then make tokenizer-sized chunks."""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_DOCUMENTS = 10
# Bound decompressed text and embedding work even for small compressed uploads.
MAX_TEXT_CHARS = 2_000_000
MAX_CHUNKS = 10_000
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


class DocumentError(ValueError):
    """A safe, actionable upload error."""


@dataclass(frozen=True)
class Section:
    text: str
    location: str


@dataclass(frozen=True)
class Chunk:
    text: str
    filename: str
    location: str
    index: int


def display_filename(filename: str) -> str:
    return Path(filename.replace("\\", "/")).name[:255] or "document"


def extract_sections(filename: str, data: bytes) -> list[Section]:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise DocumentError("Unsupported format. Upload PDF, DOCX, TXT, or Markdown.")
    if not data:
        raise DocumentError("The file is empty. Upload a document containing text.")
    if len(data) > MAX_FILE_BYTES:
        raise DocumentError("The file exceeds the 20 MB limit. Split it into smaller files.")

    sections: list[Section] = []
    total = 0

    def append(text: str, location: str) -> None:
        nonlocal total
        text = text.replace("\x00", "").strip()
        total += len(text)
        if total > MAX_TEXT_CHARS:
            raise DocumentError("Too much extracted text. Split the document into smaller files.")
        if text:
            sections.append(Section(text, location))

    try:
        if suffix == ".pdf":
            reader = PdfReader(BytesIO(data))
            if reader.is_encrypted:
                raise DocumentError("Password-protected PDFs are unsupported. Upload an unlocked PDF.")
            for number, page in enumerate(reader.pages, 1):
                append(page.extract_text() or "", f"Page {number}")
        elif suffix == ".docx":
            document = Document(BytesIO(data))
            for number, block in enumerate(document.iter_inner_content(), 1):
                if isinstance(block, Paragraph):
                    append(block.text, f"Paragraph/block {number}")
                elif isinstance(block, Table):
                    for row_number, row in enumerate(block.rows, 1):
                        append(" | ".join(cell.text for cell in row.cells),
                               f"Table/block {number}, row {row_number}")
        else:
            append(data.decode("utf-8-sig"), "Text")
    except DocumentError:
        raise
    except UnicodeDecodeError:
        raise DocumentError("Cannot decode this file. Save it as UTF-8 text and try again.") from None
    except Exception:
        raise DocumentError("Cannot read this document. Check that it is a valid, uncorrupted file.") from None
    if not sections:
        raise DocumentError("No extractable text found. Scanned PDFs require OCR before upload.")
    return sections


def chunk_sections(sections: list[Section], filename: str, tokenizer,
                   size: int = 200, overlap: int = 40) -> list[Chunk]:
    """Use the model tokenizer's offsets to preserve original text and punctuation.

    The caller supplies a tokenizer with truncation disabled; the embedding
    tokenizer itself must retain its own model input limits.
    """
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Chunk size must be positive and larger than overlap.")
    chunks = []
    for section in sections:
        encoded = tokenizer.encode(section.text, add_special_tokens=False)
        for start in range(0, len(encoded.ids), size - overlap):
            offsets = encoded.offsets[start:start + size]
            text = section.text[offsets[0][0]:offsets[-1][1]].strip()
            if text:
                chunks.append(Chunk(text, display_filename(filename), section.location, len(chunks)))
            if len(chunks) > MAX_CHUNKS:
                raise DocumentError("Too many chunks. Split the document into smaller files.")
            if start + size >= len(encoded.ids):
                break
    if not chunks:
        raise DocumentError("No indexable text found in this document.")
    return chunks
