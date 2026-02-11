# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

Outlook Email Semantic Search - A Python application that indexes Outlook emails (Microsoft 365) into a ChromaDB vector database and provides a Gradio chat interface for natural language search. Uses a hybrid retrieval system combining semantic embeddings and BM25 keyword search, with OpenAI or Ollama for conversational answers.

## Development Commands

### Setup
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Running
```bash
# Start the web interface
python app.py

# Run indexer from command line
python indexer.py          # Incremental index
python indexer.py --full   # Full reindex

# Test search from command line
python search.py "emails about budget from last month"
```

### First-Time Setup
1. Register an Azure AD app at https://portal.azure.com
2. Get the Application (Client) ID
3. Set "Allow public client flows" to Yes in Authentication settings
4. Grant `Mail.Read` API permission (Delegated)
5. For personal Microsoft accounts, set Tenant ID to `consumers`
6. Enter the Client ID in the app's Settings tab
7. Run the indexer (will prompt for Microsoft login via device code)

## Architecture

### File Structure
```
email-search/
├── app.py              # Gradio web UI with chat interface
├── indexer.py          # MS Graph API email fetching + ChromaDB indexing
├── search.py           # Hybrid semantic + BM25 search
├── config.py           # Settings management
├── requirements.txt    # Python dependencies
├── settings.json       # User settings (created on first save)
└── data/
    ├── chromadb/       # Vector database storage
    └── index_state.json # Incremental indexing state
```

### Data Flow
```
MS Graph API → indexer.py → sentence-transformers → ChromaDB
                                                        ↓
User Query → app.py → search.py → Hybrid Search → LLM → Response
                                       ↓
                            Semantic + BM25 + RRF
```

## Hybrid Retrieval System

The search uses two complementary retrieval methods combined with Reciprocal Rank Fusion (RRF).

### 1. Semantic Search (Vector Embeddings)

**How it works:**
- Each email's `Subject + Body` is converted to a 384-dimensional vector using `all-MiniLM-L6-v2`
- Query is embedded with the same model
- ChromaDB finds emails with vectors closest to query vector (cosine similarity)

**Strengths:**
- Matches by meaning: "car" finds "automobile", "budget" finds "financial planning"
- Handles paraphrasing and synonyms

**Weaknesses:**
- Poor at matching specific names, companies, or technical terms
- May miss exact keyword matches that are semantically distant

### 2. BM25 (Keyword Frequency)

**How it works:**
- All emails are tokenized into words
- BM25 index scores documents based on:
  - **Term Frequency (TF):** How often query words appear in the email
  - **Inverse Document Frequency (IDF):** How rare the words are across all emails
- Metadata (folder, sender, subject) is included in the tokenized text

**Strengths:**
- Exact keyword matching: "Original Sunshine" finds that exact phrase
- Good for names, companies, technical terms, and specific phrases

**Weaknesses:**
- No semantic understanding: "car" won't match "automobile"
- Exact words must appear in the document

### 3. Reciprocal Rank Fusion (RRF)

Combines results from both methods:

```
RRF_score = 1/(k + semantic_rank) + 1/(k + bm25_rank)
```

Where `k=60` is a constant that balances top results.

**Key insight:** Emails appearing in BOTH result lists get boosted. An email ranked #3 in both methods beats an email ranked #1 in only one method.

**Match types displayed:**
- `semantic` - Found by meaning similarity only
- `bm25` - Found by keyword match only
- `semantic+bm25` - Found by both (highest confidence)

### Retrieval Lessons Learned

1. **Semantic search alone is insufficient** for email search. Users often search for specific names, companies, or phrases that don't have semantic similarity to the email content.

2. **Folder names and metadata must be indexed** in BM25. Initially BM25 only searched email body text, missing searches like "emails in the Original Sunshine folder."

3. **BM25 index must include enriched text:**
   ```python
   enriched = f"{body} Folder: {folder} Sender: {sender} From: {sender_name}"
   ```

4. **Date filtering must be post-query** because ChromaDB's `$gte/$lte` operators don't work on string comparisons. Fetch extra results and filter in Python.

5. **Cache invalidation** - BM25 index is cached at module level. Use a version number to force rebuild when indexing logic changes.

## Key Components

**config.py**
- `Settings` dataclass with all configurable options
- `IndexState` tracks last-indexed timestamp per source
- `get_settings()` / `save_settings()` for persistence

**indexer.py**
- `MSGraphAuth` - Device code flow authentication with MSAL
- `EmailFetcher` - Paginates through all mail folders and emails
- `EmailIndexer` - Generates embeddings and upserts to ChromaDB
- `extract_text_from_attachment()` - PDF, Word, Excel, PowerPoint text extraction
- `run_indexer()` - Main entry point, supports incremental indexing

**search.py**
- `EmailSearcher` - Hybrid semantic + BM25 search with RRF
- `_get_bm25_index()` - Builds/caches BM25 index from ChromaDB data
- `format_results_for_llm()` - Prepares email context for LLM
- Uses `rank_bm25.BM25Okapi` for keyword scoring

**app.py**
- Three-tab Gradio interface: Search, Index, Settings
- Filter dropdowns for folder and date range
- `chat_response()` - Orchestrates search → LLM → response
- Supports OpenAI and Ollama backends

## Code Patterns

### Authentication
Uses MSAL device code flow - user visits a URL and enters a code. Tokens are cached by MSAL for subsequent runs. For personal Microsoft accounts, use `consumers` as tenant ID.

### Incremental Indexing
Tracks `received_datetime` of most recent email. On next run, only fetches emails newer than that timestamp. State stored in `data/index_state.json`.

### Embedding Model
Uses `all-MiniLM-L6-v2` (384 dimensions, ~80MB). Same model must be used for indexing and search to ensure compatible embeddings.

### ChromaDB
- Persistent storage in `data/chromadb/`
- Uses cosine similarity (`hnsw:space: cosine`)
- `upsert()` handles deduplication by email ID
- Metadata filtering for folder/sender works, but date comparison requires post-query filtering

### BM25 Caching
- BM25 index is built dynamically from ChromaDB data
- Cached at module level (`_bm25_cache`) for performance
- Invalidated when document count changes or `_BM25_VERSION` is incremented

## Known Limitations

- **PST files not implemented** - Only MS Graph API supported currently
- **Attachment text extraction** - Supports PDF, Word, Excel, PowerPoint, text files. Encrypted PDFs require pycryptodome.
- **Token limits** - Very long emails truncated at 8000 chars for embedding
- **Rate limits** - MS Graph API may throttle high-volume indexing
- **BM25 rebuild** - First search after restart rebuilds BM25 index (few seconds for large mailboxes)
- **Date filtering** - Done post-query, so very narrow date ranges may return fewer results
