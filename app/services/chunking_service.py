import tiktoken

CHUNK_SIZE = 800
OVERLAP = 150
STEP = CHUNK_SIZE - OVERLAP  # 650
ENCODING = "cl100k_base"


class ChunkingService:
    def __init__(self) -> None:
        self._enc = tiktoken.get_encoding(ENCODING)

    def chunk(self, pages: list[dict]) -> list[dict]:
        result = []
        for page_entry in pages:
            page = page_entry["page"]
            text = page_entry["text"]
            tokens = self._enc.encode(text)
            if not tokens:
                continue
            starts = range(0, len(tokens), STEP)
            for chunk_index, start in enumerate(starts):
                chunk_tokens = tokens[start : start + CHUNK_SIZE]
                chunk_text = self._enc.decode(chunk_tokens)
                result.append(
                    {
                        "page": page,
                        "chunk_index": chunk_index,
                        "text": chunk_text,
                        "token_count": len(chunk_tokens),
                    }
                )
        return result
