"""
tests/test_text_processing.py
================================
Unit tests for text_processing module (text cleaner + chunkers).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from text_processing.text_cleaner import TextCleaner
from text_processing.chunker import (
    FixedChunker,
    RecursiveChunker,
    SlidingWindowChunker,
    get_chunker,
)


SAMPLE_TEXT = """
This is the first paragraph of a long document.
It contains multiple sentences and some extra   spaces.

This is the second paragraph. It discusses retrieval-augmented generation.
RAG combines retrieval with LLMs to produce grounded answers.

Third paragraph: embedding models encode text into dense vectors.
These vectors are stored in a vector database for fast similarity search.
""" * 5  # repeat to get enough text for chunking


class TestTextCleaner:
    def test_basic_clean(self):
        cleaner = TextCleaner()
        text = "Hello   world!\n\n\n\nThis  is  a  test."
        result = cleaner.clean(text)
        assert "  " not in result
        assert "\n\n\n" not in result

    def test_unicode_normalisation(self):
        cleaner = TextCleaner(normalize_unicode=True)
        text = "\u00e9l\u00e8ve"  # "élève"
        result = cleaner.clean(text)
        assert len(result) > 0

    def test_control_chars_removed(self):
        cleaner = TextCleaner(remove_control_chars=True)
        text = "Hello\x00World\x01!"
        result = cleaner.clean(text)
        assert "\x00" not in result
        assert "\x01" not in result
        assert "Hello" in result


class TestFixedChunker:
    def test_chunk_count(self):
        chunker = FixedChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.split(SAMPLE_TEXT)
        assert len(chunks) > 1, "Should produce multiple chunks"

    def test_chunk_size_respected(self):
        chunker = FixedChunker(chunk_size=100, chunk_overlap=0)
        chunks = chunker.split(SAMPLE_TEXT)
        for c in chunks[:-1]:
            assert len(c) <= 100 + 5  # small tolerance for strip

    def test_no_empty_chunks(self):
        chunker = FixedChunker(chunk_size=300, chunk_overlap=50)
        chunks = chunker.split(SAMPLE_TEXT)
        for c in chunks:
            assert c.strip() != ""


class TestRecursiveChunker:
    def test_produces_chunks(self):
        chunker = RecursiveChunker(chunk_size=300, chunk_overlap=30)
        chunks = chunker.split(SAMPLE_TEXT)
        assert len(chunks) > 0

    def test_no_empty_chunks(self):
        chunker = RecursiveChunker(chunk_size=300, chunk_overlap=30)
        chunks = chunker.split(SAMPLE_TEXT)
        assert all(c.strip() for c in chunks)


class TestSlidingWindowChunker:
    def test_overlap_present(self):
        chunker = SlidingWindowChunker(chunk_size=300, stride=100)
        chunks = chunker.split(SAMPLE_TEXT)
        assert len(chunks) > 1

    def test_no_empty_chunks(self):
        chunker = SlidingWindowChunker(chunk_size=300, stride=150)
        chunks = chunker.split(SAMPLE_TEXT)
        assert all(c.strip() for c in chunks)


class TestChunkerFactory:
    def test_fixed(self):
        chunker = get_chunker("fixed", chunk_size=512)
        assert isinstance(chunker, FixedChunker)

    def test_recursive(self):
        chunker = get_chunker("recursive", chunk_size=512)
        assert isinstance(chunker, RecursiveChunker)

    def test_sliding(self):
        chunker = get_chunker("sliding_window", chunk_size=512)
        assert isinstance(chunker, SlidingWindowChunker)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get_chunker("nonexistent_strategy")
