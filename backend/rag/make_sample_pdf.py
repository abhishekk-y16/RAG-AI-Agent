from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def make_pdf(path=None):
    if path is None:
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge.pdf")
    c = canvas.Canvas(path, pagesize=letter)
    text = c.beginText(40, 750)

    lines = [
        "Insurance Agency Customer Care Knowledge Base",
        "",
        "Q: How do I file a claim?",
        "A: Call our claims line or submit through the portal. Keep photos and receipts.",
        "",
        "Q: What is a deductible?",
        "A: The amount you pay out of pocket before insurance starts paying.",
        "",
        "Q: How do I add a driver to my auto policy?",
        "A: Provide name, DOB, license number, and effective date. We'll send a quote.",
    ]

    for line in lines:
        text.textLine(line)

    c.drawText(text)
    c.showPage()
    c.save()


if __name__ == "__main__":
    make_pdf()
