# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

Outlook Email Semantic Search - A Python application that indexes Outlook emails (Microsoft 365) into a ChromaDB vector database and provides a Gradio chat interface for natural language search. Uses sentence-transformers for embeddings and supports OpenAI or Ollama for the conversational LLM.

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
3. Add `http://localhost` as a redirect URI
4. Grant `Mail.Read` API permission
5. Enter the Client ID in the app's Settings tab
6. Run the indexer (will prompt for Microsoft login)

## Architecture

### File Structure
```
email-search/
├── app.py              # Gradio web UI with chat interface
├── indexer.py          # MS Graph API email fetching + ChromaDB indexing
├── search.py           # Semantic search functions
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
User Query → app.py → search.py → ChromaDB query → LLM → Response
```

### Key Components

**config.py**
- `Settings` dataclass with all configurable options
- `IndexState` tracks last-indexed timestamp per source
- `get_settings()` / `save_settings()` for persistence

**indexer.py**
- `MSGraphAuth` - Device code flow authentication with MSAL
- `EmailFetcher` - Paginates through all mail folders and emails
- `EmailIndexer` - Generates embeddings and upserts to ChromaDB
- `run_indexer()` - Main entry point, supports incremental indexing

**search.py**
- `EmailSearcher` - Semantic search with optional filters
- `format_results_for_llm()` - Prepares email context for LLM
- `search_emails()` - Convenience function for quick searches

**app.py**
- Three-tab Gradio interface: Search, Index, Settings
- `chat_response()` - Orchestrates search → LLM → response
- Supports OpenAI and Ollama backends

## Code Patterns

### Authentication
Uses MSAL device code flow - user visits a URL and enters a code. Tokens are cached by MSAL for subsequent runs.

### Incremental Indexing
Tracks `received_datetime` of most recent email. On next run, only fetches emails newer than that timestamp. State stored in `data/index_state.json`.

### Embedding Model
Uses `all-MiniLM-L6-v2` (384 dimensions, ~80MB). Same model must be used for indexing and search to ensure compatible embeddings.

### ChromaDB
- Persistent storage in `data/chromadb/`
- Uses cosine similarity (`hnsw:space: cosine`)
- `upsert()` handles deduplication by email ID

## Known Limitations

- **PST files not implemented** - Only MS Graph API supported currently
- **No attachment indexing** - Only indexes email body text
- **Token limits** - Very long emails truncated at 8000 chars
- **Rate limits** - MS Graph API may throttle high-volume indexing
