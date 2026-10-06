"""Optional OCR adapter for reading visible broker and asset labels."""

from typing import Any


def read_text(image: Any, *, language: str = "eng") -> str:
    try:
        import pytesseract
    except ImportError as error:
        raise RuntimeError(
            "OCR requires pytesseract and the Tesseract executable."
        ) from error
    return pytesseract.image_to_string(image, lang=language).strip()
