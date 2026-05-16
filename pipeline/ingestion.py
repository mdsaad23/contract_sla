import re
from pathlib import Path


def load_contract(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Contract not found: {file_path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    else:
        return path.read_text(encoding="utf-8", errors="replace")


def _load_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def clean_text(text: str) -> str:
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)
    return text.strip()
