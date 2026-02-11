"""
Gradio web interface for email semantic search.
Provides a chat interface for natural language queries over indexed emails.
"""

import json
from typing import Optional, List, Tuple
import gradio as gr
import openai
import ollama as ollama_client

from config import get_settings, save_settings, Settings, IndexState
from search import EmailSearcher, format_results_for_llm
from indexer import run_indexer


# System prompt for the LLM
SYSTEM_PROMPT = """You are an email search assistant. You help users find and understand their emails based on natural language queries.

When answering questions:
1. Base your answers ONLY on the email excerpts provided in the context
2. Cite specific emails by mentioning the sender, date, and subject when referencing them
3. If the search results don't contain relevant information, say so clearly
4. Summarize key information from multiple emails when appropriate
5. Be concise but thorough

If asked about emails that aren't in the provided context, explain that you can only search indexed emails and suggest refining the search query."""


def get_llm_response(
    query: str,
    email_context: str,
    chat_history: List[Tuple[str, str]],
    settings: Settings,
) -> str:
    """
    Get a response from the configured LLM.

    Args:
        query: User's question
        email_context: Formatted email search results
        chat_history: Previous conversation turns
        settings: Application settings

    Returns:
        LLM's response text
    """
    # Build messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add chat history
    for user_msg, assistant_msg in chat_history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})

    # Add current query with email context
    user_content = f"""Based on the following email search results, please answer the user's question.

EMAIL SEARCH RESULTS:
{email_context}

USER QUESTION: {query}"""

    messages.append({"role": "user", "content": user_content})

    if settings.llm_provider == "openai":
        return _get_openai_response(messages, settings)
    else:
        return _get_ollama_response(messages, settings)


def _get_openai_response(messages: List[dict], settings: Settings) -> str:
    """Get response from OpenAI API."""
    if not settings.openai_api_key:
        return "Error: OpenAI API key not configured. Please set it in the Settings tab."

    try:
        client = openai.OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"OpenAI API error: {e}"


def _get_ollama_response(messages: List[dict], settings: Settings) -> str:
    """Get response from Ollama."""
    try:
        response = ollama_client.chat(
            model=settings.ollama_model,
            messages=messages,
        )
        return response["message"]["content"]
    except Exception as e:
        return f"Ollama error: {e}\n\nMake sure Ollama is running (ollama serve) and the model is pulled (ollama pull {settings.ollama_model})"


def get_available_folders() -> List[str]:
    """Get list of folders from indexed emails."""
    try:
        searcher = EmailSearcher()
        if not searcher.collection:
            return []
        # Get unique folders from metadata
        results = searcher.collection.get(include=["metadatas"], limit=10000)
        folders = set()
        if results and results.get("metadatas"):
            for meta in results["metadatas"]:
                if meta and meta.get("folder"):
                    folders.add(meta["folder"])
        return sorted(folders)
    except Exception:
        return []


def chat_response(
    message: str,
    history: List[Tuple[str, str]],
    settings_state: dict,
    folder_filter: str = "",
    date_from: str = "",
    date_to: str = "",
) -> str:
    """
    Process a chat message and return a response.

    Args:
        message: User's message
        history: Chat history
        settings_state: Current settings from state
        folder_filter: Optional folder to filter by
        date_from: Optional start date (YYYY-MM-DD)
        date_to: Optional end date (YYYY-MM-DD)

    Returns:
        Response string
    """
    # Load settings from state
    settings = Settings(**settings_state) if settings_state else get_settings()

    # Search for relevant emails with filters
    searcher = EmailSearcher(settings)
    results = searcher.search(
        message,
        n_results=settings.default_search_results,
        folder_filter=folder_filter if folder_filter else None,
        date_from=date_from if date_from else None,
        date_to=date_to if date_to else None,
    )

    # Format results for LLM context
    email_context = format_results_for_llm(results, max_emails=10)

    # Get LLM response
    response = get_llm_response(message, email_context, history, settings)

    # Add search results details for debugging
    if results:
        response += f"\n\n---\n### Search Results ({len(results)} emails found)\n"
        for i, email in enumerate(results[:10], 1):
            similarity_pct = f"{email.get('similarity', 0):.1%}"
            subject = email.get('subject', 'No subject')[:60]
            sender = email.get('sender_name') or email.get('sender', 'Unknown')
            date = email.get('received_datetime', '')[:10]
            folder = email.get('folder', '')
            match_type = email.get('match_type', 'semantic')
            bm25 = email.get('bm25_score')
            bm25_str = f", bm25: {bm25}" if bm25 else ""
            response += f"\n**{i}. [{similarity_pct}{bm25_str}] ({match_type})** {subject}\n"
            response += f"   *From:* {sender} | *Date:* {date} | *Folder:* {folder}\n"

    return response


def get_index_stats() -> str:
    """Get current indexing statistics."""
    searcher = EmailSearcher()
    stats = searcher.get_stats()
    state = IndexState.load()

    lines = [
        f"**Total Emails Indexed:** {stats['total_emails']:,}",
        "",
        "**Last Indexed:**",
    ]

    if state.last_indexed:
        for source, timestamp in state.last_indexed.items():
            count = state.email_counts.get(source, 0)
            lines.append(f"- {source}: {timestamp} ({count:,} emails)")
    else:
        lines.append("- No indexing history")

    return "\n".join(lines)


def run_index(full_reindex: bool, progress=gr.Progress()) -> str:
    """Run the email indexer."""
    try:
        progress(0, desc="Starting indexer...")
        results = run_indexer(full_reindex=full_reindex, verbose=True)

        return f"""**Indexing Complete!**

- Folders processed: {results['folders_processed']}
- Emails indexed: {results['emails_indexed']}
- Errors: {len(results['errors'])}

{get_index_stats()}"""

    except ValueError as e:
        return f"**Configuration Error:**\n\n{e}\n\nPlease configure MS Graph Client ID in the Settings tab."
    except Exception as e:
        return f"**Indexing Error:**\n\n{e}"


def load_settings_to_ui() -> Tuple:
    """Load settings and return values for UI components."""
    settings = get_settings()
    return (
        settings.ms_graph_client_id,
        settings.ms_graph_tenant_id,
        settings.llm_provider,
        settings.openai_api_key,
        settings.openai_model,
        settings.ollama_model,
        settings.ollama_base_url,
        settings.embedding_model,
        settings.default_search_results,
    )


def save_settings_from_ui(
    ms_graph_client_id: str,
    ms_graph_tenant_id: str,
    llm_provider: str,
    openai_api_key: str,
    openai_model: str,
    ollama_model: str,
    ollama_base_url: str,
    embedding_model: str,
    default_search_results: int,
) -> Tuple[str, dict]:
    """Save settings from UI and return status."""
    settings = Settings(
        ms_graph_client_id=ms_graph_client_id,
        ms_graph_tenant_id=ms_graph_tenant_id,
        llm_provider=llm_provider,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        ollama_model=ollama_model,
        ollama_base_url=ollama_base_url,
        embedding_model=embedding_model,
        default_search_results=default_search_results,
    )
    save_settings(settings)

    return "Settings saved successfully!", settings.__dict__


def create_ui() -> gr.Blocks:
    """Create the Gradio interface."""

    with gr.Blocks(title="Email Search", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Email Semantic Search")
        gr.Markdown("Search your Outlook emails using natural language queries.")

        # Hidden state for settings
        settings_state = gr.State(get_settings().__dict__)

        with gr.Tabs():
            # Chat Tab
            with gr.Tab("Search"):
                # Filter options
                with gr.Row():
                    folder_dropdown = gr.Dropdown(
                        choices=[""] + get_available_folders(),
                        value="",
                        label="Filter by Folder",
                        allow_custom_value=True,
                    )
                    date_from_input = gr.Textbox(
                        label="From Date (YYYY-MM-DD)",
                        placeholder="2024-01-01",
                        max_lines=1,
                    )
                    date_to_input = gr.Textbox(
                        label="To Date (YYYY-MM-DD)",
                        placeholder="2025-12-31",
                        max_lines=1,
                    )

                chatbot = gr.Chatbot(
                    height=400,
                )
                msg = gr.Textbox(
                    placeholder="e.g., 'What AI coding models are people recommending?'",
                    label="Your Question",
                    lines=2,
                )
                with gr.Row():
                    submit_btn = gr.Button("Search", variant="primary")
                    clear_btn = gr.Button("Clear Chat")

                def respond(message, history, settings, folder, date_from, date_to):
                    if not message.strip():
                        return history, ""

                    # Handle both tuple format and message dict format
                    history_tuples = []
                    if history:
                        if isinstance(history[0], dict):
                            # Message format - convert to tuples
                            for i in range(0, len(history) - 1, 2):
                                if history[i].get("role") == "user" and i + 1 < len(history):
                                    history_tuples.append((history[i]["content"], history[i + 1]["content"]))
                        else:
                            # Already tuple format
                            history_tuples = history

                    response = chat_response(
                        message, history_tuples, settings,
                        folder_filter=folder,
                        date_from=date_from,
                        date_to=date_to,
                    )

                    # Return in message dict format for Gradio 6
                    new_history = []
                    for user_msg, assistant_msg in history_tuples:
                        new_history.append({"role": "user", "content": user_msg})
                        new_history.append({"role": "assistant", "content": assistant_msg})
                    new_history.append({"role": "user", "content": message})
                    new_history.append({"role": "assistant", "content": response})

                    return new_history, ""

                submit_btn.click(
                    respond,
                    [msg, chatbot, settings_state, folder_dropdown, date_from_input, date_to_input],
                    [chatbot, msg],
                )
                msg.submit(
                    respond,
                    [msg, chatbot, settings_state, folder_dropdown, date_from_input, date_to_input],
                    [chatbot, msg],
                )
                clear_btn.click(lambda: ([], ""), None, [chatbot, msg])

            # Index Tab
            with gr.Tab("Index"):
                gr.Markdown("### Email Indexing")
                gr.Markdown(
                    "Index your Outlook emails to enable semantic search. "
                    "You need to configure MS Graph API credentials in Settings first."
                )

                stats_display = gr.Markdown(get_index_stats())

                with gr.Row():
                    refresh_stats_btn = gr.Button("Refresh Stats")
                    index_btn = gr.Button("Run Indexer", variant="primary")
                    full_reindex_btn = gr.Button("Full Re-index")

                index_output = gr.Markdown()

                refresh_stats_btn.click(get_index_stats, None, stats_display)
                index_btn.click(lambda: run_index(False), None, index_output)
                full_reindex_btn.click(lambda: run_index(True), None, index_output)

            # Settings Tab
            with gr.Tab("Settings"):
                gr.Markdown("### Microsoft Graph API")
                gr.Markdown(
                    "To access your Outlook emails, you need to register an Azure AD application. "
                    "[Instructions](https://docs.microsoft.com/en-us/azure/active-directory/develop/quickstart-register-app)"
                )

                ms_graph_client_id = gr.Textbox(
                    label="Client ID (Application ID)",
                    placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                )
                ms_graph_tenant_id = gr.Textbox(
                    label="Tenant ID",
                    value="common",
                    info="Use 'common' for personal Microsoft accounts, or your organization's tenant ID",
                )

                gr.Markdown("### LLM Configuration")

                llm_provider = gr.Radio(
                    choices=["ollama", "openai"],
                    label="LLM Provider",
                    value="ollama",
                )

                with gr.Group():
                    gr.Markdown("#### OpenAI Settings")
                    openai_api_key = gr.Textbox(
                        label="OpenAI API Key",
                        type="password",
                        placeholder="sk-...",
                    )
                    openai_model = gr.Textbox(
                        label="OpenAI Model",
                        value="gpt-4o-mini",
                    )

                with gr.Group():
                    gr.Markdown("#### Ollama Settings")
                    ollama_model = gr.Textbox(
                        label="Ollama Model",
                        value="llama3.2",
                        info="Run 'ollama pull llama3.2' to download",
                    )
                    ollama_base_url = gr.Textbox(
                        label="Ollama URL",
                        value="http://localhost:11434",
                    )

                gr.Markdown("### Search Settings")

                embedding_model = gr.Textbox(
                    label="Embedding Model",
                    value="all-MiniLM-L6-v2",
                    info="Sentence-transformers model for embeddings",
                )
                default_search_results = gr.Slider(
                    minimum=5,
                    maximum=50,
                    value=20,
                    step=5,
                    label="Default Search Results",
                )

                save_btn = gr.Button("Save Settings", variant="primary")
                save_status = gr.Markdown()

                # Load settings on page load
                app.load(
                    load_settings_to_ui,
                    None,
                    [
                        ms_graph_client_id,
                        ms_graph_tenant_id,
                        llm_provider,
                        openai_api_key,
                        openai_model,
                        ollama_model,
                        ollama_base_url,
                        embedding_model,
                        default_search_results,
                    ],
                )

                save_btn.click(
                    save_settings_from_ui,
                    [
                        ms_graph_client_id,
                        ms_graph_tenant_id,
                        llm_provider,
                        openai_api_key,
                        openai_model,
                        ollama_model,
                        ollama_base_url,
                        embedding_model,
                        default_search_results,
                    ],
                    [save_status, settings_state],
                )

    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
    )
