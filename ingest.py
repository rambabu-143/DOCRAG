"""
ingest.py — Robust PDF ingestion with "Golden Chunk" strategy.

Uses PyMuPDF (fitz) for reliable text extraction on memory-constrained systems.
Injects synthetic summary chunks to solve fragmentation issues for key topics.
"""

import json
import re
import os
import shutil
from pathlib import Path

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR = Path(__file__).parent
CHROMA_DIR = Path(__file__).parent / "chroma_db"
CHUNKS_FILE = Path(__file__).parent / "chunks.json"

EMBED_MODEL = "nomic-embed-text"
COLLECTION = "docrag"

# --- GOLDEN CHUNKS: Synthetic sources of truth for distributed information ---

GOLDEN_FOLDER_TYPES_CHUNK = """SDX Virtual Folder Types — Complete List
The SDX/MFT platform supports the following types of virtual folders (vFolders):
1. SDX — Standard SDX virtual folder with automatic file cleanup.
2. MBox (MBOX) — Managed File Transfer folder used by the MBOX automation engine.
3. SDX_NoFileCleanup (SDX NoFileCleanup) — Like SDX but files are NOT automatically deleted.
4. S3 (VPCx-AWS-S3) — Integration with Amazon Web Services S3 object storage buckets.
5. Azure — Integration with Microsoft Azure Blob Storage or Azure Data Lake Storage Gen2.
6. GCP (GCP Bucket) — Integration with Google Cloud Platform storage buckets.
7. Yandex S3 — Yandex object storage (S3-compatible API); configured same as S3.
8. HDFS (Hadoop) — Integration with Enterprise Data Lake EDL-Hadoop HDFS file system.
9. Exchange (MEA / O365 IMAP) — Extracts file attachments from Exchange/O365 mailboxes using IMAP or basic auth.
10. Exchange OAuth2 — Modern O365 mailbox integration using Microsoft Graph API and OAuth2.
11. SharePoint — Legacy integration with Microsoft SharePoint Online (being retired April 2026).
12. SharePointOauth2 (Entra App) — Replacement SharePoint integration using MS Entra app + certificate.
13. Box — Integration with Box.com accounts using JWT (JSON Web Token) authentication.

Note: SDX, MBox, and SDX_NoFileCleanup types are interchangeable. Cloud types (S3, Azure, HDFS, GCP, Box, SharePoint) cannot be swapped once configured."""

GOLDEN_SFTP_PROCEDURE_CHUNK = """Step-by-Step Guide: How to Transfer Files from SFTP (MBOX)
Based on the MBOX Administration Guide (Section 6.2), follow these steps to configure SFTP transfers:

1. Create an SFTP Partner:
   - In MBOX Admin GUI, go to Partners tab -> Add button.
   - Host Protocol: Select 'SFTP'.
   - Hostname: Enter the FQDN (e.g., sftp.partner.com). Do not use IP addresses.
   - Authentication: Provide Username and Password (Basic) or SSH2 Public Keys.

2. Configure "Pull" Transfers (Extracting files from remote SFTP):
   - On the Partner settings, go to 'Pull from' configuration.
   - Source Folder: Specify the directory on the remote SFTP server.
   - Archive Folder (Mandatory): Specify where MBOX moves successfully processed files to prevent duplicates.
   - Send & Pull Interval: Set to at least 600 seconds to prevent platform overload.

3. Configure "Push" Transfers (Uploading files to remote SFTP):
   - On the Partner settings, go to 'Send to' configuration.
   - Target Folder: Specify the destination directory on the remote server.
   - Temp Folder (Best Practice): MBOX uploads to temp first, then renames to target to avoid partial file processing.

4. Link to an Interface:
   - Create an Interface (Class: Pull or Push) with a filename pattern (e.g., *.txt).
   - Link the Interface to your SFTP Partner."""


def is_noise(text: str) -> bool:
    """Filter out garbled PDF artifacts/metadata blocks."""
    stripped = text.strip()
    if not stripped or len(stripped) < 20:
        return True
    
    digit_chars = sum(c.isdigit() for c in stripped)
    # Garbled table/screenshot dumps: very high digit ratio
    if digit_chars / max(len(stripped), 1) > 0.4:
        return True
    
    # Generic footer/header noise
    if re.search(r"Page \d+ of \d+|Confidential|Internal Use Only", stripped, re.I):
        if len(stripped) < 60:
            return True
            
    return False

def load_pdfs() -> list[Document]:
    documents = []
    if not DOCS_DIR.exists():
        print(f"Error: {DOCS_DIR} not found.")
        return []

    pdf_files = list(DOCS_DIR.glob("*.pdf"))
    for pdf_path in pdf_files:
        print(f"  Processing: {pdf_path.name}")
        try:
            doc = fitz.open(str(pdf_path))
            for page_num, page in enumerate(doc):
                text = page.get_text()
                
                # Simple section detection: look for first line in bold/large or just first line
                lines = text.strip().split("\n")
                section = lines[0][:100] if lines else "General"
                
                if not is_noise(text):
                    documents.append(Document(
                        page_content=text,
                        metadata={
                            "source": pdf_path.name,
                            "page": page_num + 1,
                            "section": section
                        }
                    ))
            doc.close()
        except Exception as e:
            print(f"  × Error loading {pdf_path.name}: {e}")
            
    return documents

def ingest():
    print("\n────────────── DocRAG Ingest ──────────────\n")
    
    # 1. Load and Split
    docs = load_pdfs()
    if not docs:
        print("No documents found. Skipping.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = splitter.split_documents(docs)
    
    # 2. Inject GOLDEN CHUNKS
    print("  Injecting Golden Procedural Chunks...")
    chunks.insert(0, Document(
        page_content=GOLDEN_FOLDER_TYPES_CHUNK,
        metadata={
            "source": "SDX-UI Admin Guide_new.pdf",
            "section": "SDX Virtual Folder Types Summary",
            "page": 0,
            "_synthetic": True
        }
    ))
    chunks.insert(1, Document(
        page_content=GOLDEN_SFTP_PROCEDURE_CHUNK,
        metadata={
            "source": "MBOX4_Admin.pdf",
            "section": "SFTP Transfer Setup Guide",
            "page": 0,
            "_synthetic": True
        }
    ))

    # 3. Save to JSON (for BM25/inspection)
    CHUNKS_FILE.write_text(json.dumps(
        [{"page_content": c.page_content, "metadata": c.metadata} for c in chunks],
        indent=2
    ))

    # 4. Vectorize
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    
    print(f"  Vectorizing {len(chunks)} chunks...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION
    )

    print(f"\n  ✓ {len(chunks)} chunks indexed.")
    print("  Run uv run main.py query \n")

if __name__ == "__main__":
    ingest()
