"""
Configuration management for the email search application.
Handles settings persistence and loading.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, List
from pathlib import Path

# Project root directory
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
SETTINGS_FILE = PROJECT_DIR / "settings.json"
INDEX_STATE_FILE = DATA_DIR / "index_state.json"

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)


@dataclass
class Settings:
    """Application settings with defaults."""

    # Microsoft Graph API settings
    ms_graph_client_id: str = ""
    ms_graph_tenant_id: str = "consumers"  # "consumers" for personal accounts, or specific tenant ID

    # LLM settings
    llm_provider: str = "ollama"  # "openai" or "ollama"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    ollama_model: str = "llama3.2"
    ollama_base_url: str = "http://localhost:11434"

    # Embedding model
    embedding_model: str = "all-MiniLM-L6-v2"

    # PST file paths (optional)
    pst_file_paths: List[str] = field(default_factory=list)

    # ChromaDB settings
    chroma_collection_name: str = "emails"

    # Search settings
    default_search_results: int = 20

    def save(self, path: Optional[Path] = None) -> None:
        """Save settings to JSON file."""
        save_path = path or SETTINGS_FILE
        with open(save_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Settings":
        """Load settings from JSON file, or return defaults if not found."""
        load_path = path or SETTINGS_FILE
        if load_path.exists():
            try:
                with open(load_path, 'r') as f:
                    data = json.load(f)
                return cls(**data)
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Warning: Could not load settings from {load_path}: {e}")
                return cls()
        return cls()


@dataclass
class IndexState:
    """Tracks indexing state for incremental updates."""

    # Last indexed timestamp per source (ISO format)
    # Key: source identifier (e.g., "msgraph" or PST file path)
    # Value: ISO timestamp of last indexed email
    last_indexed: dict = field(default_factory=dict)

    # Total emails indexed per source
    email_counts: dict = field(default_factory=dict)

    def save(self) -> None:
        """Save index state to JSON file."""
        with open(INDEX_STATE_FILE, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls) -> "IndexState":
        """Load index state from JSON file, or return empty state if not found."""
        if INDEX_STATE_FILE.exists():
            try:
                with open(INDEX_STATE_FILE, 'r') as f:
                    data = json.load(f)
                return cls(**data)
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Warning: Could not load index state: {e}")
                return cls()
        return cls()

    def update_source(self, source: str, last_timestamp: str, count: int) -> None:
        """Update the state for a specific source."""
        self.last_indexed[source] = last_timestamp
        self.email_counts[source] = count
        self.save()


# Global settings instance (lazy loaded)
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance, loading from disk if needed."""
    global _settings
    if _settings is None:
        _settings = Settings.load()
    return _settings


def save_settings(settings: Settings) -> None:
    """Save settings and update global instance."""
    global _settings
    settings.save()
    _settings = settings


if __name__ == "__main__":
    # Test settings save/load
    settings = get_settings()
    print("Current settings:")
    print(json.dumps(asdict(settings), indent=2))

    # Save defaults if no settings file exists
    if not SETTINGS_FILE.exists():
        settings.save()
        print(f"\nCreated default settings file at: {SETTINGS_FILE}")
