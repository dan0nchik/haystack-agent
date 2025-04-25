import os
from pathlib import Path
from haystack import Pipeline
from haystack.dataclasses import Document, ChatMessage
from haystack.utils.auth import Secret

# 🔌 Haystack components
from haystack.components.routers import FileTypeRouter
from haystack.components.converters import MarkdownToDocument
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.writers import DocumentWriter
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator

# 🧠 Qdrant integration
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever

# 🔐 Keys
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or input("OpenAI API Key: ")

# 📁 Obsidian Vault
vault_path = Path("./obsidian").expanduser()
md_files = list(vault_path.glob("**/*.md"))

# 🔧 Qdrant setup (disk-based, change URL for cloud)
document_store = QdrantDocumentStore(
    ":memory:",
    index="Document",
    embedding_dim=384,
    recreate_index=True,
    similarity="cosine",
    return_embedding=True,
)

# ========================
# 1. Indexing Pipeline
# ========================
index_pipeline = Pipeline()

index_pipeline.add_component("router", FileTypeRouter(mime_types=["text/markdown"]))
index_pipeline.add_component("md_converter", MarkdownToDocument())
index_pipeline.add_component("joiner", DocumentJoiner())
index_pipeline.add_component("cleaner", DocumentCleaner())
index_pipeline.add_component(
    "splitter", DocumentSplitter(split_by="word", split_length=150, split_overlap=50)
)
index_pipeline.add_component(
    "embedder",
    SentenceTransformersDocumentEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    ),
)
index_pipeline.add_component("writer", DocumentWriter(document_store=document_store))

index_pipeline.connect("router.text/markdown", "md_converter.sources")
index_pipeline.connect("md_converter", "joiner")
index_pipeline.connect("joiner", "cleaner")
index_pipeline.connect("cleaner", "splitter")
index_pipeline.connect("splitter", "embedder")
index_pipeline.connect("embedder", "writer")

# Run indexing
index_pipeline.run({"router": {"sources": md_files}})

# ========================
# 2. RAG Query Pipeline
# ========================
query_pipeline = Pipeline()

# 🔍 Query embedder & retriever
query_pipeline.add_component(
    "text_embedder",
    SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"),
)
query_pipeline.add_component(
    "retriever", QdrantEmbeddingRetriever(document_store=document_store)
)

# 📄 Prompt Template
template = [
    ChatMessage.from_user(
        """
You are a helpful assistant. Use the following context from my notes to answer the question.

Context:
{% for document in documents %}
{{ document.content }}
{% endfor %}

Question: {{ question }}
Answer:
"""
    )
]
query_pipeline.add_component("prompt_builder", ChatPromptBuilder(template=template))

# 🤖 OpenAI LLM
query_pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o"))

# Connect components
query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
query_pipeline.connect("retriever.documents", "prompt_builder.documents")
query_pipeline.connect("prompt_builder.prompt", "llm.messages")


# ========================
# 3. Ask a question
# ========================
def ask_note_question(question: str):
    result = query_pipeline.run(
        {"text_embedder": {"text": question}, "prompt_builder": {"question": question}}
    )
    return result["llm"]["replies"][0].text


# Example
print(ask_note_question("Summarize my notes about business."))
