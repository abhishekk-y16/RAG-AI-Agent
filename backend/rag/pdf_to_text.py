from pypdf import PdfReader


def pdf_to_text(pdf_path: str) -> tuple:
    """Extract text and page count from PDF.
    Returns: (text, page_count)
    """
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")

    text = "\n".join(pages)
    text = text.replace("\r", "\n")
    text = "\n".join([line.strip() for line in text.split("\n") if line.strip()])
    return text, len(reader.pages)
