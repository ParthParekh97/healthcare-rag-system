import os
import glob
from dataclasses import dataclass, field
from typing import Optional

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None


@dataclass
class Document:
    content: str
    metadata: dict = field(default_factory=dict)
    doc_id: str = ""
    source: str = ""


class DocumentLoader:
    """Loads documents from a directory. Supports .txt and .pdf files."""

    SUPPORTED_EXTENSIONS = {".txt", ".pdf"}

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_all(self) -> list[Document]:
        documents = []
        for ext in self.SUPPORTED_EXTENSIONS:
            pattern = os.path.join(self.data_dir, f"*{ext}")
            for filepath in sorted(glob.glob(pattern)):
                doc = self._load_file(filepath)
                if doc:
                    documents.append(doc)
        return documents

    def _load_file(self, filepath: str) -> Optional[Document]:
        filename = os.path.basename(filepath)
        ext = os.path.splitext(filepath)[1].lower()

        if ext == ".txt":
            return self._load_txt(filepath, filename)
        elif ext == ".pdf":
            return self._load_pdf(filepath, filename)
        return None

    def _load_txt(self, filepath: str, filename: str) -> Document:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        metadata = self._extract_metadata(content)
        metadata["filename"] = filename
        metadata["filepath"] = filepath

        return Document(
            content=content,
            metadata=metadata,
            doc_id=metadata.get("document_id", filename),
            source=filepath,
        )

    def _load_pdf(self, filepath: str, filename: str) -> Optional[Document]:
        if PdfReader is None:
            print(f"Skipping {filename}: PyPDF2 not installed")
            return None

        reader = PdfReader(filepath)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())

        content = "\n\n".join(pages)
        if not content.strip():
            return None

        return Document(
            content=content,
            metadata={"filename": filename, "filepath": filepath, "pages": len(reader.pages)},
            doc_id=filename,
            source=filepath,
        )

    def _extract_metadata(self, content: str) -> dict:
        metadata = {}
        for line in content.split("\n")[:5]:
            line = line.strip()
            if line.startswith("Title:"):
                metadata["title"] = line[len("Title:"):].strip()
            elif line.startswith("Source:"):
                metadata["source_org"] = line[len("Source:"):].strip()
            elif line.startswith("Document ID:"):
                metadata["document_id"] = line[len("Document ID:"):].strip()
        return metadata
