"""Unit tests for structure-aware chunker."""

from __future__ import annotations

import subprocess
import sys
import pytest

from studyos.chunking.chunker import StructureAwareChunker, estimate_token_count
from studyos.ingestion.models import PageElement, ParsedDocument


class TestStructureAwareChunker:
    def test_short_element_intact_single_chunk(self):
        """Short elements should remain intact as single chunks without artificial padding."""
        chunker = StructureAwareChunker(target_chunk_size=300, chunk_overlap=50)
        el = PageElement(
            text="Short note on binary search time complexity.",
            location_type="page",
            location_index=1,
            location_title="Complexity",
        )
        doc = ParsedDocument(
            filename="notes.pdf",
            file_type="pdf",
            sha256_hash="hash123",
            file_size_bytes=500,
            elements=[el],
        )

        chunks = chunker.chunk_document(doc, course_id="crs_test", document_id="doc_test")
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.chunk_index == 0
        assert chunk.location_index == 1
        assert chunk.location_type == "page"
        assert "[Source: notes.pdf | Page: 1 | Section: Complexity]" in chunk.content
        assert "Short note on binary search" in chunk.content

    def test_long_element_split_with_overlap(self):
        """Long elements exceeding the target budget must be split across multiple chunks with overlap."""
        chunker = StructureAwareChunker(target_chunk_size=50, chunk_overlap=15)
        # Create a paragraph with several sentences
        sentences = [
            f"Sentence number {i} explains a distinct algorithmic concept in detail."
            for i in range(1, 15)
        ]
        long_text = " ".join(sentences)

        el = PageElement(
            text=long_text,
            location_type="section",
            location_index=1,
            location_title="Long Algorithm Section",
        )
        doc = ParsedDocument(
            filename="algo.txt",
            file_type="txt",
            sha256_hash="hash456",
            file_size_bytes=2000,
            elements=[el],
        )

        chunks = chunker.chunk_document(doc, course_id="crs_algo", document_id="doc_algo")
        assert len(chunks) > 1

        # Check that consecutive chunks have overlap
        first_chunk_text = chunks[0].content
        second_chunk_text = chunks[1].content

        # Both chunks must have the breadcrumb header
        assert "[Source: algo.txt | Section: 1" in first_chunk_text
        assert "[Source: algo.txt | Section: 1" in second_chunk_text

        # Every chunk index must be sequential
        for idx, chk in enumerate(chunks):
            assert chk.chunk_index == idx
            assert chk.document_id == "doc_algo"
            assert chk.course_id == "crs_algo"

    def test_source_boundary_strict_isolation(self):
        """
        Chunks must NEVER merge text across distinct PDF pages or PPTX slides,
        even if individual pages/slides are short.
        """
        chunker = StructureAwareChunker(target_chunk_size=500, chunk_overlap=50)

        el_slide1 = PageElement(
            text="Slide 1: Dynamic Programming definition.",
            location_type="slide",
            location_index=1,
            location_title="DP Intro",
        )
        el_slide2 = PageElement(
            text="Slide 2: Memoization and Tabulation differences.",
            location_type="slide",
            location_index=2,
            location_title="Memo vs Tab",
        )

        doc = ParsedDocument(
            filename="lecture.pptx",
            file_type="pptx",
            sha256_hash="hash789",
            file_size_bytes=1000,
            elements=[el_slide1, el_slide2],
        )

        chunks = chunker.chunk_document(doc, course_id="crs_cs", document_id="doc_cs")
        assert len(chunks) == 2

        # Chunk 0 corresponds strictly to Slide 1
        assert chunks[0].location_type == "slide"
        assert chunks[0].location_index == 1
        assert "Dynamic Programming definition" in chunks[0].content
        assert "Memoization" not in chunks[0].content

        # Chunk 1 corresponds strictly to Slide 2
        assert chunks[1].location_type == "slide"
        assert chunks[1].location_index == 2
        assert "Memoization and Tabulation" in chunks[1].content
        assert "DP Intro" not in chunks[1].content

    def test_oversized_sentence_word_splitting(self):
        """A single unusually long sentence exceeding the chunk budget must split safely without error."""
        chunker = StructureAwareChunker(target_chunk_size=40, chunk_overlap=10)
        # Create a single sentence with 100 words (no period until the very end)
        long_sentence = " ".join([f"word{i}" for i in range(100)]) + "."

        el = PageElement(
            text=long_sentence,
            location_type="page",
            location_index=1,
            location_title="Giant Sentence",
        )
        doc = ParsedDocument(
            filename="oversized.pdf",
            file_type="pdf",
            sha256_hash="hash999",
            file_size_bytes=1500,
            elements=[el],
        )

        chunks = chunker.chunk_document(doc, course_id="crs_1", document_id="doc_1")
        assert len(chunks) >= 2
        # Ensure all words are present across chunks
        joined = " ".join(c.content for c in chunks)
        assert "word0" in joined
        assert "word99" in joined

    def test_unicode_and_punctuation_handling(self):
        """Chunker must handle unicode characters and mathematical formulas deterministically."""
        chunker = StructureAwareChunker()
        text = "Formula: ∀x ∈ S, f(x) ≥ 0. Complex eigenvalues: λ₁ = 2 + 3i, λ₂ = 2 - 3i. German: Größe, Maß."
        el = PageElement(
            text=text,
            location_type="section",
            location_index=1,
            location_title="Math",
        )
        doc = ParsedDocument(
            filename="math.md",
            file_type="txt",
            sha256_hash="hashmath",
            file_size_bytes=500,
            elements=[el],
        )

        chunks = chunker.chunk_document(doc, course_id="crs_math", document_id="doc_math")
        assert len(chunks) == 1
        assert "∀x ∈ S" in chunks[0].content
        assert "Größe" in chunks[0].content
        assert chunks[0].content_hash is not None

    def test_deterministic_token_estimation(self):
        t1 = estimate_token_count("Hello world!")
        t2 = estimate_token_count("Hello world!")
        assert t1 == t2
        assert t1 == 3  # 'Hello', 'world', '!'
        assert estimate_token_count("") == 0

    def test_hang_regression_with_subprocess_timeout(self):
        """
        Regression test for original hang: run the problematic parameter combination
        in a subprocess with a strict 5-second timeout. If it hangs, fail with diagnostic.
        """
        script = """
import sys
from studyos.chunking.chunker import StructureAwareChunker
from studyos.ingestion.models import PageElement

chunker = StructureAwareChunker(
    target_chunk_size=40,
    chunk_overlap=25,
    min_chunk_size=20,
)
# Two medium-length sentences (~16 and ~14 tokens, exceeding body budget of 26)
s1 = "Algorithm analysis evaluates the execution time and memory resources required by an arbitrary computer program."
s2 = "Computational complexity theory formally classifies algorithmic problems based on their inherent mathematical difficulty."
el = PageElement(
    text=f"{s1} {s2}",
    location_type="page",
    location_index=1,
    location_title="Hang Reproduction",
)
chunks = chunker.chunk_element(
    element=el,
    filename="lecture.pdf",
    course_id="crs_test",
    document_id="doc_test",
    start_chunk_index=0,
)
assert len(chunks) == 2, f"Expected 2 chunks, got {len(chunks)}"
assert "Algorithm analysis" in chunks[0].content
assert "Computational complexity" in chunks[1].content
sys.exit(0)
"""
        try:
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except subprocess.TimeoutExpired as exc:
            pytest.fail(f"Chunker hung in infinite loop and timed out after 5 seconds: {exc}")

        assert result.returncode == 0, f"Subprocess failed with code {result.returncode}:\n{result.stderr}"

    def test_exact_hang_parameter_combination_terminates(self):
        """
        In-process test for exact parameter combination (40 target, 25 overlap, 20 min).
        Verifies that when overlap + next sentence > body_budget, overlap is discarded
        rather than repeatedly emitted as identical overlap-only chunks.
        """
        chunker = StructureAwareChunker(
            target_chunk_size=40,
            chunk_overlap=25,
            min_chunk_size=20,
        )
        s1 = "Algorithm analysis evaluates the execution time and memory resources required by an arbitrary computer program."
        s2 = "Computational complexity theory formally classifies algorithmic problems based on their inherent mathematical difficulty."
        el = PageElement(
            text=f"{s1} {s2}",
            location_type="page",
            location_index=1,
            location_title="Hang Reproduction",
        )
        chunks = chunker.chunk_element(
            element=el,
            filename="lecture.pdf",
            course_id="crs_test",
            document_id="doc_test",
            start_chunk_index=0,
        )

        assert len(chunks) == 2
        assert chunks[0].chunk_index == 0
        assert chunks[1].chunk_index == 1

        # Chunk 0 contains sentence 1
        assert "Algorithm analysis" in chunks[0].content
        # Chunk 1 contains sentence 2; overlap was cleanly discarded because S1 + S2 > budget
        assert "Computational complexity" in chunks[1].content
        assert "Algorithm analysis" not in chunks[1].content

        # Source metadata preserved on both
        assert "[Source: lecture.pdf | Page: 1 | Section: Hang Reproduction]" in chunks[0].content
        assert "[Source: lecture.pdf | Page: 1 | Section: Hang Reproduction]" in chunks[1].content

    def test_incoming_sentence_cannot_fit_alongside_retained_overlap(self):
        """
        When Sentence 3 cannot fit alongside the retained overlap of Sentence 2,
        overlap is safely trimmed/discarded and Sentence 3 starts a clean new chunk.
        """
        chunker = StructureAwareChunker(
            target_chunk_size=40,
            chunk_overlap=20,
            min_chunk_size=20,
        )
        # Sentence 1: ~10 tokens, Sentence 2: ~10 tokens, Sentence 3: ~22 tokens
        s1 = "Binary search divides the search interval in half repeatedly."
        s2 = "Linear search scans every element sequentially in worst-case time."
        s3 = "Asymptotic notation formally describes upper bounds, tight bounds, and lower bounds on runtime growth as problem size approaches infinity."

        el = PageElement(
            text=f"{s1} {s2} {s3}",
            location_type="page",
            location_index=1,
            location_title="Search",
        )
        doc = ParsedDocument(
            filename="search.pdf",
            file_type="pdf",
            sha256_hash="hash_search",
            file_size_bytes=1000,
            elements=[el],
        )

        chunks = chunker.chunk_document(doc, course_id="crs_test", document_id="doc_test")
        assert len(chunks) >= 2
        # Verify all sentences appear and no content is lost
        combined = " ".join(c.content for c in chunks)
        assert s1 in combined
        assert s2 in combined
        assert s3 in combined

    def test_oversized_sentence_with_surrounding_sentences(self):
        """
        An oversized sentence surrounded by normal sentences must flush preceding sentences,
        decompose the oversized sentence with word-level fallback, and resume normally.
        """
        chunker = StructureAwareChunker(target_chunk_size=40, chunk_overlap=10, min_chunk_size=20)
        s1 = "This is a normal leading sentence before the oversized sentence."
        s2 = " ".join([f"Giantword{i}" for i in range(80)]) + "."
        s3 = "This is a normal trailing sentence after the oversized sentence."

        el = PageElement(
            text=f"{s1} {s2} {s3}",
            location_type="section",
            location_index=1,
            location_title="Oversized Test",
        )
        doc = ParsedDocument(
            filename="giant.txt",
            file_type="txt",
            sha256_hash="hash_giant",
            file_size_bytes=2000,
            elements=[el],
        )

        chunks = chunker.chunk_document(doc, course_id="crs_test", document_id="doc_test")
        assert len(chunks) >= 3

        # Chunk 0 has leading sentence
        assert "normal leading sentence" in chunks[0].content

        # Middle chunks have giant words
        combined_middle = " ".join(c.content for c in chunks[1:-1])
        assert "Giantword0" in combined_middle
        assert "Giantword79" in combined_middle

        # Final chunk has trailing sentence
        assert "normal trailing sentence" in chunks[-1].content

    def test_content_preservation_and_deterministic_output(self):
        """Chunking the same document twice must produce identical chunks, hashes, and token counts."""
        chunker = StructureAwareChunker(target_chunk_size=60, chunk_overlap=20, min_chunk_size=25)
        text = (
            "Graph algorithms explore vertices and edges. "
            "Dijkstra solves single-source shortest path problems. "
            "Bellman-Ford detects negative weight cycles in graphs. "
            "Floyd-Warshall computes all-pairs shortest paths."
        )
        el = PageElement(
            text=text,
            location_type="slide",
            location_index=1,
            location_title="Graph Algorithms",
        )
        doc = ParsedDocument(
            filename="graphs.pptx",
            file_type="pptx",
            sha256_hash="hash_graphs",
            file_size_bytes=1200,
            elements=[el],
        )

        run1 = chunker.chunk_document(doc, course_id="crs_g", document_id="doc_g")
        run2 = chunker.chunk_document(doc, course_id="crs_g", document_id="doc_g")

        assert len(run1) == len(run2)
        for c1, c2 in zip(run1, run2):
            assert c1.content == c2.content
            assert c1.content_hash == c2.content_hash
            assert c1.token_count == c2.token_count
            assert c1.chunk_index == c2.chunk_index
            assert c1.location_type == c2.location_type
            assert c1.location_index == c2.location_index
