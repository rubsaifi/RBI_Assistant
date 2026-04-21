"""
PDF Loader Module
Handles loading and processing of RBI Master PDF documents.
"""

import os
from pathlib import Path
from typing import List, Dict
import re

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None


def clean_text(text: str) -> str:
    """Clean and normalize extracted text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\;\:\-\(\)\[\]\/]', '', text)
    # Remove extra newlines
    text = re.sub(r'\n+', '\n', text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks for better retrieval.

    Args:
        text: The text to split
        chunk_size: Size of each chunk
        overlap: Overlap between chunks

    Returns:
        List of text chunks
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        # Try to end at a sentence boundary
        if end < len(text):
            # Look for sentence endings
            sentence_end = text.rfind('.', end - 100, end)
            if sentence_end != -1:
                end = sentence_end + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


def extract_text_pdfplumber(pdf_path: str) -> str:
    """Extract text using pdfplumber (more accurate)."""
    if pdfplumber is None:
        raise ImportError("pdfplumber not installed")

    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"

    return text


def extract_text_pypdf2(pdf_path: str) -> str:
    """Extract text using PyPDF2 (fallback)."""
    if PdfReader is None:
        raise ImportError("PyPDF2 not installed")

    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        text += page.extract_text() + "\n\n"

    return text


def load_and_process_pdf(pdf_path: str) -> List[Dict]:
    """
    Load PDF and process it into document chunks.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of document dictionaries with text and metadata
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Try pdfplumber first, then PyPDF2
    try:
        raw_text = extract_text_pdfplumber(pdf_path)
        extraction_method = "pdfplumber"
    except Exception as e:
        print(f"pdfplumber failed, trying PyPDF2: {e}")
        raw_text = extract_text_pypdf2(pdf_path)
        extraction_method = "pypdf2"

    # Clean the text
    cleaned_text = clean_text(raw_text)

    # Split into chunks
    chunks = chunk_text(cleaned_text)

    # Create document objects
    documents = []
    for i, chunk in enumerate(chunks):
        documents.append({
            "content": chunk,
            "metadata": {
                "source": pdf_path,
                "chunk_index": i,
                "extraction_method": extraction_method,
                "total_chunks": len(chunks)
            }
        })

    return documents


def get_document_stats(documents: List[Dict]) -> Dict:
    """Get statistics about loaded documents."""
    total_chunks = len(documents)
    total_chars = sum(len(doc["content"]) for doc in documents)
    avg_chunk_size = total_chars / total_chunks if total_chunks > 0 else 0

    return {
        "total_chunks": total_chunks,
        "total_characters": total_chars,
        "avg_chunk_size": round(avg_chunk_size, 2)
    }


if __name__ == "__main__":
    # Test loading
    test_path = "../Docs/rbi_master.pdf"
    if os.path.exists(test_path):
        docs = load_and_process_pdf(test_path)
        stats = get_document_stats(docs)
        print(f"Loaded {stats['total_chunks']} chunks")
        print(f"Total characters: {stats['total_characters']}")
        print(f"Average chunk size: {stats['avg_chunk_size']}")
