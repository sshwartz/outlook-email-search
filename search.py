"""
Semantic + BM25 hybrid search for indexed emails.
"""

import json
import logging
import re
from typing import Optional, List, Dict, Any
import chromadb

# Suppress model loading warnings
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore")
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from config import get_settings, DATA_DIR, Settings

# Module-level BM25 cache (persists across EmailSearcher instances in same process)
_bm25_cache: Optional[Dict] = None
_BM25_VERSION = 2  # Increment to force cache rebuild


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r'\b\w+\b', text.lower())


class EmailSearcher:
    """Hybrid semantic + BM25 search over indexed emails."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

        # Initialize embedding model (same as indexer for consistent embeddings)
        self.embedder = SentenceTransformer(self.settings.embedding_model)

        # Connect to ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=str(DATA_DIR / "chromadb"),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        try:
            self.collection = self.chroma_client.get_collection(
                name=self.settings.chroma_collection_name,
            )
        except Exception:
            self.collection = None

    def _get_bm25_index(self) -> Optional[Dict]:
        """Get or build BM25 index (cached at module level)."""
        global _bm25_cache

        if not self.collection:
            return None

        current_count = self.collection.count()

        # Return cached index if it exists and hasn't changed
        if (_bm25_cache
            and _bm25_cache.get("count") == current_count
            and _bm25_cache.get("version") == _BM25_VERSION):
            return _bm25_cache

        # Build BM25 index from all documents
        print("Building BM25 index...")
        batch_size = 5000
        all_ids = []
        all_docs = []
        all_metadatas = []
        offset = 0

        while True:
            batch = self.collection.get(
                include=["documents", "metadatas"],
                limit=batch_size,
                offset=offset,
            )
            if not batch or not batch["ids"]:
                break
            all_ids.extend(batch["ids"])
            all_docs.extend(batch["documents"])
            all_metadatas.extend(batch["metadatas"])
            offset += len(batch["ids"])
            if len(batch["ids"]) < batch_size:
                break

        # Tokenize documents enriched with metadata (folder, sender, recipients)
        enriched_docs = []
        for doc, meta in zip(all_docs, all_metadatas):
            parts = [doc or ""]
            if meta:
                if meta.get("folder"):
                    parts.append(f"Folder: {meta['folder']}")
                if meta.get("sender"):
                    parts.append(f"Sender: {meta['sender']}")
                if meta.get("sender_name"):
                    parts.append(f"From: {meta['sender_name']}")
                if meta.get("subject"):
                    parts.append(f"Subject: {meta['subject']}")
            enriched_docs.append(" ".join(parts))

        tokenized = [_tokenize(doc) for doc in enriched_docs]

        # Build BM25 index
        bm25 = BM25Okapi(tokenized)

        _bm25_cache = {
            "bm25": bm25,
            "ids": all_ids,
            "docs": all_docs,
            "metadatas": all_metadatas,
            "count": current_count,
            "version": _BM25_VERSION,
        }
        print(f"BM25 index built with {current_count} documents")
        return _bm25_cache

    def search(
        self,
        query: str,
        n_results: int = 20,
        sender_filter: Optional[str] = None,
        folder_filter: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining semantic similarity and BM25 keyword matching.

        Uses Reciprocal Rank Fusion (RRF) to merge results from both methods.
        """
        if not self.collection:
            return []

        # Build where clause for ChromaDB metadata filtering
        where_clauses = []
        if sender_filter:
            where_clauses.append({"sender": {"$contains": sender_filter.lower()}})
        if folder_filter:
            where_clauses.append({"folder": folder_filter})

        where = None
        if len(where_clauses) == 1:
            where = where_clauses[0]
        elif len(where_clauses) > 1:
            where = {"$and": where_clauses}

        # Over-fetch for post-query date filtering
        fetch_count = n_results * 3

        # --- 1. Semantic search ---
        query_embedding = self.embedder.encode([query])[0].tolist()
        semantic_ranks = {}

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=fetch_count,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
            if results and results["ids"] and results["ids"][0]:
                for rank, (eid, meta, doc, dist) in enumerate(zip(
                    results["ids"][0],
                    results["metadatas"][0],
                    results["documents"][0],
                    results["distances"][0],
                )):
                    semantic_ranks[eid] = {
                        "rank": rank,
                        "distance": dist,
                        "metadata": meta or {},
                        "document": doc or "",
                    }
        except Exception as e:
            print(f"Semantic search error: {e}")

        # --- 2. BM25 search ---
        bm25_ranks = {}
        bm25_cache = self._get_bm25_index()

        if bm25_cache:
            query_tokens = _tokenize(query)
            scores = bm25_cache["bm25"].get_scores(query_tokens)

            # Get top results by BM25 score
            scored_indices = sorted(
                enumerate(scores), key=lambda x: x[1], reverse=True
            )

            rank = 0
            for idx, score in scored_indices:
                if score <= 0:
                    break
                if rank >= fetch_count:
                    break

                eid = bm25_cache["ids"][idx]
                meta = bm25_cache["metadatas"][idx] or {}

                # Apply metadata filters
                if folder_filter and meta.get("folder") != folder_filter:
                    continue
                if sender_filter and sender_filter.lower() not in (meta.get("sender") or "").lower():
                    continue

                bm25_ranks[eid] = {
                    "rank": rank,
                    "bm25_score": score,
                    "metadata": meta,
                    "document": bm25_cache["docs"][idx] or "",
                }
                rank += 1

        # --- 3. Reciprocal Rank Fusion ---
        k = 60  # RRF constant
        rrf_scores = {}
        all_ids = set(semantic_ranks.keys()) | set(bm25_ranks.keys())

        for eid in all_ids:
            score = 0
            source = []

            if eid in semantic_ranks:
                score += 1.0 / (k + semantic_ranks[eid]["rank"])
                source.append("semantic")

            if eid in bm25_ranks:
                score += 1.0 / (k + bm25_ranks[eid]["rank"])
                source.append("bm25")

            # Get metadata and document from whichever source has it
            data = semantic_ranks.get(eid) or bm25_ranks.get(eid)

            rrf_scores[eid] = {
                "rrf_score": score,
                "metadata": data["metadata"],
                "document": data["document"],
                "semantic_dist": semantic_ranks[eid]["distance"] if eid in semantic_ranks else None,
                "bm25_score": bm25_ranks[eid]["bm25_score"] if eid in bm25_ranks else None,
                "source": "+".join(source),
            }

        # Sort by RRF score (higher = better)
        sorted_results = sorted(
            rrf_scores.items(), key=lambda x: x[1]["rrf_score"], reverse=True
        )

        # --- 4. Format results with date filtering ---
        emails = []
        for email_id, data in sorted_results:
            metadata = data["metadata"]
            received = metadata.get("received_datetime", "") or ""

            # Apply date filters
            if date_from and received and received < date_from:
                continue
            if date_to and received and received > date_to:
                continue

            # Compute display similarity from semantic distance if available
            if data["semantic_dist"] is not None:
                similarity = 1 - data["semantic_dist"]
            else:
                similarity = 0  # BM25-only match

            # Parse recipient JSON
            to_recipients = []
            if metadata.get("to"):
                try:
                    to_recipients = json.loads(metadata["to"])
                except json.JSONDecodeError:
                    pass

            emails.append({
                "id": email_id,
                "subject": metadata.get("subject", ""),
                "sender": metadata.get("sender", ""),
                "sender_name": metadata.get("sender_name", ""),
                "to": to_recipients,
                "received_datetime": received,
                "folder": metadata.get("folder", ""),
                "content_preview": self._get_preview(data["document"], 300),
                "similarity": round(similarity, 4),
                "rrf_score": round(data["rrf_score"], 6),
                "bm25_score": round(data["bm25_score"], 2) if data["bm25_score"] else None,
                "match_type": data["source"],
            })

            if len(emails) >= n_results:
                break

        return emails

    def _get_preview(self, text: str, max_length: int = 300) -> str:
        """Extract a preview snippet from email content."""
        if text.startswith("Subject:"):
            lines = text.split("\n", 1)
            text = lines[1] if len(lines) > 1 else ""

        text = text.strip()

        if len(text) <= max_length:
            return text

        truncated = text[:max_length]
        last_space = truncated.rfind(" ")
        if last_space > max_length * 0.8:
            truncated = truncated[:last_space]

        return truncated + "..."

    def get_email_by_id(self, email_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific email by ID."""
        if not self.collection:
            return None

        try:
            result = self.collection.get(
                ids=[email_id],
                include=["documents", "metadatas"],
            )

            if result and result["ids"]:
                metadata = result["metadatas"][0] if result["metadatas"] else {}
                document = result["documents"][0] if result["documents"] else ""

                to_recipients = []
                if metadata.get("to"):
                    try:
                        to_recipients = json.loads(metadata["to"])
                    except json.JSONDecodeError:
                        pass

                return {
                    "id": email_id,
                    "subject": metadata.get("subject", ""),
                    "sender": metadata.get("sender", ""),
                    "sender_name": metadata.get("sender_name", ""),
                    "to": to_recipients,
                    "received_datetime": metadata.get("received_datetime", ""),
                    "folder": metadata.get("folder", ""),
                    "content": document,
                }
        except Exception as e:
            print(f"Error retrieving email {email_id}: {e}")

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        if not self.collection:
            return {"total_emails": 0, "indexed": False}

        return {
            "total_emails": self.collection.count(),
            "indexed": True,
        }


def search_emails(
    query: str,
    n_results: int = 20,
    **filters,
) -> List[Dict[str, Any]]:
    """Convenience function for searching emails."""
    searcher = EmailSearcher()
    return searcher.search(query, n_results=n_results, **filters)


def format_email_for_llm(email: Dict[str, Any], include_content: bool = True) -> str:
    """Format an email result for inclusion in LLM context."""
    lines = [
        f"From: {email.get('sender_name', '')} <{email.get('sender', '')}>",
        f"Subject: {email.get('subject', '')}",
        f"Date: {email.get('received_datetime', '')}",
        f"Folder: {email.get('folder', '')}",
    ]

    if include_content and email.get("content_preview"):
        lines.append(f"Preview: {email['content_preview']}")

    return "\n".join(lines)


def format_results_for_llm(emails: List[Dict[str, Any]], max_emails: int = 10) -> str:
    """Format search results for LLM context."""
    if not emails:
        return "No emails found matching your query."

    formatted = []
    for i, email in enumerate(emails[:max_emails], 1):
        formatted.append(f"--- Email {i} (similarity: {email.get('similarity', 0):.2%}) ---")
        formatted.append(format_email_for_llm(email))
        formatted.append("")

    return "\n".join(formatted)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python search.py <query>")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    print(f"Searching for: {query}\n")

    results = search_emails(query, n_results=5)

    if results:
        for i, email in enumerate(results, 1):
            print(f"{i}. [{email['match_type']}] (sim: {email['similarity']:.1%}, bm25: {email.get('bm25_score', 'N/A')}) {email['subject'][:60]}")
            print(f"   From: {email['sender_name'] or email['sender']} | {email['received_datetime'][:10]} | {email['folder']}")
            print()
    else:
        print("No results found. Make sure you've run the indexer first.")
