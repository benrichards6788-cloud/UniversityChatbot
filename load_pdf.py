
import os
import sys
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from tqdm import tqdm #progress bar

INPUT_DIR = Path("guidance pdf")      
OUTPUT_DIR = Path("rawpdf")     
MERGED_OUTPUT = OUTPUT_DIR / "ALL_TEXT_MERGED.txt"  # mega-file

def extract_pdf_to_text(pdf_path: Path) -> str:
    """return full text from a PDF with page separators."""
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    chunks = []
    for i, page in enumerate(pages, start=1):
        chunks.append(f"\n\n--- PAGE {i} ---\n\n{page.page_content}")
    return "".join(chunks).strip()

def main(merge_all: bool = True):
    if not INPUT_DIR.exists():
        print(f"Input folder not found: {INPUT_DIR.resolve()}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted([p for p in INPUT_DIR.iterdir() if p.suffix.lower() == ".pdf"])
    if not pdf_files:
        print(f"No PDFs found in {INPUT_DIR.resolve()}")
        sys.exit(0)

    merged_parts = []
    errors = []

    for pdf in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            text = extract_pdf_to_text(pdf)
            out_path = OUTPUT_DIR / (pdf.stem + ".txt")
            out_path.write_text(text, encoding="utf-8")
            if merge_all:
                merged_parts.append(f"\n\n===== FILE: {pdf.name} =====\n\n{text}")
        except Exception as e:
            errors.append((pdf.name, str(e)))

    if merge_all and merged_parts:
        MERGED_OUTPUT.write_text("".join(merged_parts).strip(), encoding="utf-8")

    print(f"\nDone. Wrote {len(pdf_files) - len(errors)} text files to {OUTPUT_DIR.resolve()}.")
    if merge_all:
        print(f"Merged file: {MERGED_OUTPUT.resolve()}")
    if errors:
        print("\nSome files failed:")
        for name, msg in errors:
            print(f" - {name}: {msg}")

if __name__ == "__main__":
    merge = "--no-merge" not in sys.argv
    main(merge_all=merge)
