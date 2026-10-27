"""Structure-aware document chunking engine for StudyOS."""

from __future__ import annotations

import re
from studyos.ingestion.models import DocumentChunk, PageElement, ParsedDocument


# Common technical and academic abbreviations that should not trigger sentence splits
COMMON_ABBREVIATIONS = {
    "e.g.",
    "i.e.",
    "etc.",
    "vs.",
    "fig.",
    "dr.",
    "prof.",
    "mr.",
    "mrs.",
    "ms.",
    "al.",
    "approx.",
    "no.",
    "vol.",
    "p.",
    "pp.",
}


def estimate_token_count(text: str) -> int:
    """
    Estimate the number of tokens in text using deterministic token-boundary heuristics.

    Note on Strategy & Limitations:
    This estimator counts alphanumeric word tokens and individual punctuation marks.
    In technical and mathematical prose, this closely approximates subword tokenizers
    (such as Byte-Pair Encoding in tiktoken or WordPiece in BERT) with an average of
    1.1–1.3 tokens per whitespace-separated word, or roughly 4 characters per token.
    This local estimation does not guarantee exact token equivalence with future
    multimodal or specialized embedding models, but provides a deterministic,
    lightweight proxy for chunk sizing.
    """
    if not text:
        return 0
    tokens = re.findall(r"\w+|[^\w\s]", text)
    return max(1, len(tokens))


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences while respecting common technical abbreviations."""
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    sentences: list[str] = []

    for para in paragraphs:
        # Split on sentence delimiters (. ? !) followed by whitespace and an uppercase letter/digit/quote
        parts = re.split(r"([.?!])\s+(?=[A-Z0-9\"'(\[])", para)
        if len(parts) <= 1:
            sentences.append(para)
            continue

        current = ""
        i = 0
        while i < len(parts):
            seg = parts[i]
            punct = parts[i + 1] if i + 1 < len(parts) else ""
            candidate = f"{current}{seg}{punct}".strip()

            words = candidate.split()
            last_word = words[-1].lower() if words else ""
            if last_word in COMMON_ABBREVIATIONS and i + 2 < len(parts):
                current = f"{candidate} "
                i += 2
            else:
                if candidate:
                    sentences.append(candidate)
                current = ""
                i += 2

        if current.strip():
            sentences.append(current.strip())

    return sentences if sentences else [text.strip()]


class StructureAwareChunker:
    """
    Splits ParsedDocument elements into bounded, semantically cohesive chunks.

    Guarantees:
    - Never merges text across disparate PDF pages or PPTX slides.
    - Prepends an evidence-tracking breadcrumb header to every chunk.
    - Preserves sentence boundaries within elements where practical.
    - Keeps short slides or sections intact without artificial padding.
    - Enforces configurable token size targets and overlap.
    """

    def __init__(
        self,
        target_chunk_size: int = 500,
        chunk_overlap: int = 75,
        min_chunk_size: int = 50,
    ) -> None:
        if chunk_overlap >= target_chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than target_chunk_size ({target_chunk_size})"
            )
        self.target_chunk_size = target_chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def _build_header(
        self,
        filename: str,
        location_type: str,
        location_index: int,
        location_title: str | None,
    ) -> str:
        """Construct the breadcrumb header according to docs/ARCHITECTURE.md specification."""
        loc_label = location_type.capitalize()
        header = f"[Source: {filename} | {loc_label}: {location_index}"
        if location_title:
            header += f" | Section: {location_title}"
        header += "]"
        return header

    def chunk_element(
        self,
        element: PageElement,
        filename: str,
        course_id: str,
        document_id: str,
        start_chunk_index: int,
    ) -> list[DocumentChunk]:
        """Chunk a single PageElement without crossing element boundaries."""
        text = element.text.strip()
        if not text:
            return []

        header = self._build_header(
            filename=filename,
            location_type=element.location_type,
            location_index=element.location_index,
            location_title=element.location_title,
        )
        header_tokens = estimate_token_count(header)

        # Budget remaining for body text
        body_budget = max(self.min_chunk_size, self.target_chunk_size - header_tokens)

        element_tokens = estimate_token_count(text)
        # If element fits comfortably in a single chunk, preserve it intact!
        if element_tokens <= body_budget:
            full_content = f"{header}\n{text}"
            chunk = DocumentChunk.create(
                document_id=document_id,
                course_id=course_id,
                chunk_index=start_chunk_index,
                location_type=element.location_type,
                location_index=element.location_index,
                location_title=element.location_title,
                content=full_content,
                token_count=estimate_token_count(full_content),
            )
            return [chunk]

        # Element exceeds budget -> decompose into bounded sentence-preserving windows
        sentences = split_into_sentences(text)
        chunks: list[DocumentChunk] = []
        current_sentences: list[str] = []
        current_token_count = 0
        current_idx = start_chunk_index

        i = 0
        while i < len(sentences):
            prev_i = i
            sentence = sentences[i]
            sent_tokens = estimate_token_count(sentence)

            # If a single sentence exceeds body_budget, split it by words
            if sent_tokens > body_budget:
                # Flush accumulated sentences first to preserve document order
                if current_sentences:
                    chunk_text = " ".join(current_sentences)
                    full_content = f"{header}\n{chunk_text}"
                    chunks.append(
                        DocumentChunk.create(
                            document_id=document_id,
                            course_id=course_id,
                            chunk_index=current_idx,
                            location_type=element.location_type,
                            location_index=element.location_index,
                            location_title=element.location_title,
                            content=full_content,
                            token_count=estimate_token_count(full_content),
                        )
                    )
                    current_idx += 1
                    current_sentences = []
                    current_token_count = 0

                words = sentence.split()
                word_buffer: list[str] = []
                word_tokens = 0
                for w in words:
                    wt = estimate_token_count(w)
                    if word_tokens + wt > body_budget and word_buffer:
                        part_text = " ".join(word_buffer)
                        full_content = f"{header}\n{part_text}"
                        chunks.append(
                            DocumentChunk.create(
                                document_id=document_id,
                                course_id=course_id,
                                chunk_index=current_idx,
                                location_type=element.location_type,
                                location_index=element.location_index,
                                location_title=element.location_title,
                                content=full_content,
                                token_count=estimate_token_count(full_content),
                            )
                        )
                        current_idx += 1
                        word_buffer = []
                        word_tokens = 0
                    word_buffer.append(w)
                    word_tokens += wt

                if word_buffer:
                    part_text = " ".join(word_buffer)
                    full_content = f"{header}\n{part_text}"
                    chunks.append(
                        DocumentChunk.create(
                            document_id=document_id,
                            course_id=course_id,
                            chunk_index=current_idx,
                            location_type=element.location_type,
                            location_index=element.location_index,
                            location_title=element.location_title,
                            content=full_content,
                            token_count=estimate_token_count(full_content),
                        )
                    )
                    current_idx += 1

                i += 1
                assert i == prev_i + 1, "Loop progress invariant violated"
                continue

            if current_token_count + sent_tokens <= body_budget:
                current_sentences.append(sentence)
                current_token_count += sent_tokens
                i += 1
            else:
                # Flush current window
                if current_sentences:
                    chunk_text = " ".join(current_sentences)
                    full_content = f"{header}\n{chunk_text}"
                    chunks.append(
                        DocumentChunk.create(
                            document_id=document_id,
                            course_id=course_id,
                            chunk_index=current_idx,
                            location_type=element.location_type,
                            location_index=element.location_index,
                            location_title=element.location_title,
                            content=full_content,
                            token_count=estimate_token_count(full_content),
                        )
                    )
                    current_idx += 1

                # Calculate overlap: backtrack sentences that can fit alongside the incoming sentence
                overlap_tokens = 0
                overlap_sentences: list[str] = []
                back_idx = len(current_sentences) - 1
                while back_idx >= 0 and overlap_tokens < self.chunk_overlap:
                    back_sent = current_sentences[back_idx]
                    bst = estimate_token_count(back_sent)
                    # Sentence must fit within chunk_overlap AND allow the next sentence to fit in body_budget
                    if (
                        overlap_tokens + bst <= self.chunk_overlap
                        and overlap_tokens + bst + sent_tokens <= body_budget
                    ):
                        overlap_sentences.insert(0, back_sent)
                        overlap_tokens += bst
                    else:
                        break
                    back_idx -= 1

                # Begin new window with trimmed overlap and the incoming sentence
                current_sentences = list(overlap_sentences) + [sentence]
                current_token_count = overlap_tokens + sent_tokens
                i += 1

            assert i == prev_i + 1, "Loop progress invariant violated"

        # Flush remaining sentences
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            full_content = f"{header}\n{chunk_text}"
            chunks.append(
                DocumentChunk.create(
                    document_id=document_id,
                    course_id=course_id,
                    chunk_index=current_idx,
                    location_type=element.location_type,
                    location_index=element.location_index,
                    location_title=element.location_title,
                    content=full_content,
                    token_count=estimate_token_count(full_content),
                )
            )

        return chunks

    def chunk_document(
        self,
        parsed_doc: ParsedDocument,
        course_id: str,
        document_id: str,
    ) -> list[DocumentChunk]:
        """Chunk all elements of a ParsedDocument in sequential order."""
        all_chunks: list[DocumentChunk] = []
        chunk_idx = 0

        for element in parsed_doc.elements:
            element_chunks = self.chunk_element(
                element=element,
                filename=parsed_doc.filename,
                course_id=course_id,
                document_id=document_id,
                start_chunk_index=chunk_idx,
            )
            all_chunks.extend(element_chunks)
            chunk_idx += len(element_chunks)

        return all_chunks
