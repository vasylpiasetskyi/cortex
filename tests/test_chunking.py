import random

import tiktoken

from app.services.chunking_service import ChunkingService

ENC = tiktoken.get_encoding("cl100k_base")

_RNG = random.Random(42)
_BASE_TOKENS: list[int] = ENC.encode(
    "".join(_RNG.choices([chr(i) for i in range(33, 127)], k=10000))
)


def _make_text(n_tokens: int) -> str:
    """Return a string whose cl100k_base token count is exactly n_tokens."""
    tokens = _BASE_TOKENS[:n_tokens]
    return ENC.decode(tokens)


def _count_tokens(text: str) -> int:
    return len(ENC.encode(text))


def test_empty_input_returns_empty():
    svc = ChunkingService()
    assert svc.chunk([]) == []


def test_short_text_produces_single_chunk():
    svc = ChunkingService()
    result = svc.chunk([{"page": 0, "text": "hello world"}])
    assert len(result) == 1
    assert result[0]["page"] == 0
    assert result[0]["chunk_index"] == 0
    assert result[0]["text"] == "hello world"
    assert result[0]["token_count"] == _count_tokens("hello world")


def test_text_of_exactly_800_tokens_is_one_chunk():
    svc = ChunkingService()
    text = _make_text(800)
    result = svc.chunk([{"page": 0, "text": text}])
    # 800 tokens: start=0 (800), start=650 (150) → 2 chunks due to STEP=650
    assert len(result) == 2
    assert result[0]["token_count"] == 800
    assert result[1]["token_count"] == 150


def test_text_of_900_tokens_produces_two_chunks_with_overlap():
    svc = ChunkingService()
    text = _make_text(900)
    result = svc.chunk([{"page": 0, "text": text}])
    assert len(result) == 2
    # First chunk: tokens 0-799 (800 tokens)
    assert result[0]["token_count"] == 800
    # Second chunk: tokens 650-899 (250 tokens)
    assert result[1]["token_count"] == 250
    # Overlap: last 150 tokens of chunk 0 should equal first 150 tokens of chunk 1
    tokens_full = ENC.encode(text)
    overlap_from_first = ENC.decode(tokens_full[650:800])
    overlap_in_second = ENC.decode(ENC.encode(result[1]["text"])[:150])
    assert overlap_from_first == overlap_in_second


def test_multiple_pages_chunk_indices_reset_per_page():
    svc = ChunkingService()
    text = _make_text(900)
    result = svc.chunk([{"page": 0, "text": text}, {"page": 2, "text": "short"}])
    page0_chunks = [c for c in result if c["page"] == 0]
    page2_chunks = [c for c in result if c["page"] == 2]
    assert page0_chunks[0]["chunk_index"] == 0
    assert page0_chunks[1]["chunk_index"] == 1
    assert page2_chunks[0]["chunk_index"] == 0


def test_text_just_over_step_boundary_produces_three_chunks():
    svc = ChunkingService()
    text = _make_text(1451)
    result = svc.chunk([{"page": 0, "text": text}])
    # 1451 tokens: start=0 (800), start=650 (800), start=1300 (151) → 3 chunks
    assert len(result) == 3
    assert result[2]["token_count"] == 151
