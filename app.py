#!/usr/bin/env python3
"""
Simple Gradio Frontend for RAG System

A clean interface to chat with either Simple RAG or Agentic RAG,
with a dedicated tab to view retrieved documents.
"""

import gradio as gr
from typing import Tuple, List, Dict
import time
from loguru import logger

# Import RAG components
from config_loader import get_config
from search_utils import SearchEngine
from simple_rag import SimpleRAG
from agentic_rag import AgenticRAG


class RAGChatApp:
    def __init__(self):
        """Initialize the RAG Chat application."""
        self.config = get_config()
        self.search_engine = None
        self.simple_rag = None
        self.agentic_rag = None
        self.last_retrieved_docs = ""

        # Initialize components
        self._initialize_components()

        logger.info("🚀 RAG Chat App initialized successfully!")

    def _initialize_components(self):
        """Initialize search engine and RAG models."""
        try:
            # Create search engine
            self.search_engine = SearchEngine(
                chroma_path=self.config.get("paths.chroma_path"),
                bm25_index_path=self.config.get("paths.bm25_index"),
                bm25_metadata_path=self.config.get("paths.bm25_metadata"),
                embedding_model_name=self.config.get("models.embedding.name"),
                reranker_model_name=self.config.get("models.reranker.name"),
                reciprocal_rank_k=self.config.get("search.reciprocal_rank_k")
            )

            # Initialize Simple RAG
            self.simple_rag = SimpleRAG(
                model_name=self.config.get("rag.simple.default_model"),
                temperature=self.config.get("rag.simple.default_temperature"),
                top_k=self.config.get("rag.simple.default_top_k"),
                search_engine=self.search_engine,
                system_prompt=self.config.get("prompts.simple_rag_system")
            )

            # Initialize Agentic RAG
            self.agentic_rag = AgenticRAG(
                model_name=self.config.get("rag.agentic.default_model"),
                temperature=self.config.get("rag.agentic.default_temperature"),
                search_engine=self.search_engine,
                grade_prompt=self.config.get("prompts.grade"),
                rewrite_prompt=self.config.get("prompts.rewrite"),
                generate_prompt=self.config.get("prompts.generate")
            )

            logger.success("✅ All RAG components initialized!")

        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            raise

    def chat_with_rag(
        self,
        message: str,
        history: List[Tuple[str, str]],
        rag_mode: str,
        top_k: int
    ) -> Tuple[List[Tuple[str, str]], str, str]:
        """
        Process user message with selected RAG mode.

        Returns:
            - Updated chat history
            - Empty string (for clearing input)
            - Retrieved documents
        """
        if not message.strip():
            return history, "", self.last_retrieved_docs

        logger.info(f"🔄 Processing query with {rag_mode}: {message[:50]}...")
        start_time = time.time()

        try:
            # Get search results first (for display purposes)
            search_results = self.search_engine.hybrid_search(message, top_k=top_k)
            self.last_retrieved_docs = self._format_retrieved_docs(search_results, top_k)

            # Process with selected RAG mode
            if rag_mode == "Simple RAG":
                # Create fresh Simple RAG with updated top_k
                simple_rag = SimpleRAG(
                    model_name=self.config.get("rag.simple.default_model"),
                    temperature=self.config.get("rag.simple.default_temperature"),
                    top_k=top_k,
                    search_engine=self.search_engine,
                    system_prompt=self.config.get("prompts.simple_rag_system")
                )
                response = simple_rag.query(message)
            else:  # Agentic RAG
                response = self.agentic_rag.query(message)

            # Update chat history
            history.append((message, response))

            elapsed_time = time.time() - start_time
            logger.success(f"✅ Response generated in {elapsed_time:.2f}s with {rag_mode}")

        except Exception as e:
            error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
            logger.error(f"Error processing query: {e}")
            history.append((message, error_msg))
            self.last_retrieved_docs = "Error occurred during search."

        return history, "", self.last_retrieved_docs

    def _format_retrieved_docs(self, search_results: str, top_k: int) -> str:
        """Format retrieved documents for display."""
        if not search_results:
            return "No documents retrieved."

        documents = search_results.split('\n\n')

        formatted_docs = f"**📚 Retrieved Documents (Top {min(len(documents), top_k)}):**\n\n"

        for i, doc in enumerate(documents[:top_k], 1):
            # Truncate very long documents
            display_doc = doc[:500] + "..." if len(doc) > 500 else doc
            formatted_docs += f"**Document {i}:**\n{display_doc}\n\n---\n\n"

        return formatted_docs

    def clear_chat(self) -> Tuple[List, str]:
        """Clear chat history and retrieved docs."""
        self.last_retrieved_docs = ""
        logger.info("🧹 Chat history cleared")
        return [], ""

    def create_interface(self) -> gr.Interface:
        """Create and return the Gradio interface."""

        # Custom CSS for better styling
        css = """
        .gradio-container {
            max-width: 1200px !important;
        }
        .chat-container {
            height: 500px;
        }
        .docs-container {
            height: 500px;
            overflow-y: auto;
        }
        """

        with gr.Blocks(css=css, title="RAG Chat System", theme=gr.themes.Soft()) as interface:

            gr.Markdown(
                """
                # 🤖 RAG Chat System

                Chat with your documents using either **Simple RAG** or **Agentic RAG**.
                The Agentic RAG can refine queries and re-search if needed, while Simple RAG provides direct responses.
                """
            )

            with gr.Row():
                with gr.Column(scale=2):
                    # Chat Interface
                    gr.Markdown("### 💬 Chat Interface")

                    chatbot = gr.Chatbot(
                        value=[],
                        elem_classes=["chat-container"],
                        show_label=False,
                    )

                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Ask me anything about your documents...",
                            show_label=False,
                            scale=4
                        )
                        send_btn = gr.Button("Send 📤", scale=1, variant="primary")

                    # Controls
                    with gr.Row():
                        rag_mode = gr.Radio(
                            choices=["Simple RAG", "Agentic RAG"],
                            value="Simple RAG",
                            label="RAG Mode",
                            scale=2
                        )
                        top_k = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=1,
                            label="Documents to Retrieve",
                            scale=1
                        )
                        clear_btn = gr.Button("Clear Chat 🧹", scale=1)

                with gr.Column(scale=1):
                    # Retrieved Documents Display
                    gr.Markdown("### 📚 Retrieved Documents")

                    docs_display = gr.Markdown(
                        value="No documents retrieved yet. Start chatting to see retrieved content!",
                        elem_classes=["docs-container"]
                    )

            # Event handlers
            def handle_chat(message, history, mode, k):
                return self.chat_with_rag(message, history, mode, k)

            def handle_clear():
                return self.clear_chat()

            # Wire up events
            msg_input.submit(
                fn=handle_chat,
                inputs=[msg_input, chatbot, rag_mode, top_k],
                outputs=[chatbot, msg_input, docs_display]
            )

            send_btn.click(
                fn=handle_chat,
                inputs=[msg_input, chatbot, rag_mode, top_k],
                outputs=[chatbot, msg_input, docs_display]
            )

            clear_btn.click(
                fn=handle_clear,
                outputs=[chatbot, docs_display]
            )

            # Footer
            gr.Markdown(
                """
                ---
                **💡 Tips:**
                - Try both RAG modes to compare responses
                - Agentic RAG may take longer but can provide more refined answers
                - Adjust the number of documents to retrieve based on your query complexity
                - Check the Retrieved Documents panel to see what sources were found
                """
            )

        return interface


def main():
    """Main application entry point."""
    logger.info("🚀 Starting RAG Chat Application...")

    try:
        # Initialize the app
        app = RAGChatApp()

        # Create and launch interface
        interface = app.create_interface()

        # Launch with custom settings
        interface.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            show_error=True,
        )

    except Exception as e:
        logger.error(f"❌ Failed to start application: {e}")
        raise


if __name__ == "__main__":
    main()