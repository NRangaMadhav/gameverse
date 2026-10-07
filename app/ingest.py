import argparse
import logging
from pathlib import Path

from app.settings import DEFAULT_PDF_DIR
from app.vector_store import ingest_pdfs

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Index GameVerse India PDF research sources.")
    parser.add_argument(
        "--directory",
        default=str(DEFAULT_PDF_DIR),
        help="Directory containing PDF files (defaults to the project's pdfs folder).",
    )
    arguments = parser.parse_args()
    result = ingest_pdfs(Path(arguments.directory))
    print(
        f"Indexed {result['files']} PDF(s): {result['documents']} extracted items, "
        f"{result['chunks']} vector chunks."
    )


if __name__ == "__main__":
    main()
