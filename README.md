# Outlook Email Semantic Search

A Python application that indexes your Outlook emails into a vector database and provides a chat interface for natural language search. Uses hybrid retrieval combining semantic embeddings and BM25 keyword search for accurate results.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

- **Hybrid Search** - Combines semantic similarity (meaning) with BM25 (keywords) using Reciprocal Rank Fusion
- **Natural Language Queries** - Ask questions like "What did John say about the Q4 budget?"
- **LLM-Powered Answers** - Uses OpenAI or Ollama to summarize and answer based on search results
- **Attachment Indexing** - Extracts text from PDF, Word, Excel, and PowerPoint attachments
- **Folder & Date Filtering** - Filter results by mail folder and date range
- **Incremental Indexing** - Only fetches new emails after initial sync
- **Web Interface** - Clean Gradio UI with chat, indexing controls, and settings

## How It Works

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│  Outlook Email  │────▶│   Indexer    │────▶│  ChromaDB   │
│  (MS Graph API) │     │              │     │  (Vectors)  │
└─────────────────┘     │  + Extract   │     └──────┬──────┘
                        │  Attachments │            │
                        └──────────────┘            │
                                                    ▼
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│   Your Query    │────▶│Hybrid Search │◀────│    BM25     │
│                 │     │  (RRF Fusion)│     │   (Keywords)│
└─────────────────┘     └──────┬───────┘     └─────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │  LLM Answer  │
                        │ (OpenAI/     │
                        │  Ollama)     │
                        └──────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
git clone https://github.com/sshwartz/outlook-email-search.git
cd outlook-email-search
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### 2. Register Azure AD App

1. Go to [Azure Portal](https://portal.azure.com) → Azure Active Directory → App registrations
2. Click **New registration**
3. Name: `Email Search` (or anything)
4. Supported account types: Choose based on your account type
5. Click **Register**
6. Copy the **Application (client) ID**
7. Go to **Authentication** → Set "Allow public client flows" to **Yes**
8. Go to **API permissions** → Add **Microsoft Graph** → **Mail.Read** (Delegated)

### 3. Run the App

```bash
python app.py
```

Open http://localhost:7861 in your browser.

### 4. Configure & Index

1. Go to **Settings** tab
2. Paste your **Client ID**
3. For personal Microsoft accounts, set Tenant ID to `consumers`
4. Choose LLM provider (Ollama recommended for privacy)
5. Click **Save Settings**
6. Go to **Index** tab → Click **Run Indexer**
7. Follow the device code login prompt in your terminal

### 5. Search

Go to the **Search** tab and ask questions like:
- "Show me emails about the project deadline"
- "What did Sarah say about the marketing budget?"
- "Find emails with PDF attachments about contracts"

## Search Methods

The app uses two complementary search methods:

| Method | Matches | Example |
|--------|---------|---------|
| **Semantic** | Meaning & concepts | "car" finds "automobile" |
| **BM25** | Exact keywords | "John Smith" finds that exact name |

Results are combined using **Reciprocal Rank Fusion (RRF)** - emails appearing in both result sets get boosted.

## Configuration

Settings are stored in `settings.json`:

| Setting | Description |
|---------|-------------|
| `ms_graph_client_id` | Azure AD Application ID |
| `ms_graph_tenant_id` | `consumers` for personal, `common` for work accounts |
| `llm_provider` | `ollama` or `openai` |
| `ollama_model` | e.g., `llama3.2` |
| `openai_api_key` | Your OpenAI API key |
| `embedding_model` | Default: `all-MiniLM-L6-v2` |

## Command Line Usage

```bash
# Run indexer directly
python indexer.py          # Incremental (new emails only)
python indexer.py --full   # Full reindex

# Test search
python search.py "emails about budget"
```

## Requirements

- Python 3.9+
- Outlook account (personal or Microsoft 365)
- For Ollama: [Install Ollama](https://ollama.com) and run `ollama pull llama3.2`
- For OpenAI: API key with credits

## Privacy

- All data stays local in `data/` directory
- Embeddings are generated locally (no API calls)
- Only the LLM step sends data externally (use Ollama for full privacy)
- `settings.json` contains your API keys - don't commit it

## License

MIT
