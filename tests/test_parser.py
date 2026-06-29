from unittest.mock import MagicMock, patch

from app.services.parser_service import ParserService


def make_mock_pdf(page_texts: list[str]):
    """Build a fake pypdf.PdfReader with pages that return given texts."""
    mock_reader = MagicMock()
    pages = []
    for text in page_texts:
        page = MagicMock()
        page.extract_text.return_value = text
        pages.append(page)
    mock_reader.pages = pages
    return mock_reader


def test_two_pages_returns_both():
    svc = ParserService()
    with patch(
        "app.services.parser_service.PdfReader",
        return_value=make_mock_pdf(["Page one text", "Page two text"]),
    ):
        result = svc.parse("dummy.pdf")
    assert len(result) == 2
    assert result[0] == {"page": 0, "text": "Page one text"}
    assert result[1] == {"page": 1, "text": "Page two text"}


def test_empty_pages_are_skipped():
    svc = ParserService()
    with patch(
        "app.services.parser_service.PdfReader",
        return_value=make_mock_pdf(["text", "", "   ", "more text"]),
    ):
        result = svc.parse("dummy.pdf")
    assert len(result) == 2
    assert result[0]["page"] == 0
    assert result[1]["page"] == 3
