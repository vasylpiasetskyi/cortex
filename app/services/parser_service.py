from pypdf import PdfReader


class ParserService:
    def parse(self, file_path: str) -> list[dict]:
        reader = PdfReader(file_path)
        result = []
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                result.append({"page": page_num, "text": text})
        return result
