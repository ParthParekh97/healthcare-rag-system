from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    chunk_id: str = ""
    doc_id: str = ""


class TextChunker:
    """Splits documents into overlapping chunks for embedding."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, content: str, doc_id: str = "", metadata: dict = None) -> list[Chunk]:
        metadata = metadata or {}
        sections = self._split_by_sections(content)
        chunks = []

        for section_title, section_text in sections:
            section_chunks = self._split_text(section_text)
            for i, chunk_text in enumerate(section_chunks):
                chunk_meta = {**metadata}
                if section_title:
                    chunk_meta["section"] = section_title

                chunk = Chunk(
                    text=chunk_text.strip(),
                    metadata=chunk_meta,
                    chunk_id=f"{doc_id}_chunk_{len(chunks)}",
                    doc_id=doc_id,
                )
                chunks.append(chunk)

        return chunks

    def _split_by_sections(self, text: str) -> list[tuple[str, str]]:
        lines = text.split("\n")
        sections = []
        current_title = ""
        current_lines = []

        for line in lines:
            stripped = line.strip()
            if self._is_section_header(stripped):
                if current_lines:
                    sections.append((current_title, "\n".join(current_lines)))
                current_title = stripped
                current_lines = [line]
            else:
                current_lines.append(line)

        if current_lines:
            sections.append((current_title, "\n".join(current_lines)))

        return sections if sections else [("", text)]

    def _is_section_header(self, line: str) -> bool:
        if not line:
            return False
        if len(line) < 3 or len(line) > 120:
            return False
        if line[0].isdigit() and ". " in line[:5]:
            return True
        return False

    def _split_text(self, text: str) -> list[str]:
        words = text.split()
        if len(words) <= self.chunk_size:
            return [text] if text.strip() else []

        chunks = []
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start = end - self.chunk_overlap

        return chunks
