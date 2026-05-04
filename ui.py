import gradio as gr
import anthropic
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.anthropic import Anthropic
from llama_index.core import Settings
from dotenv import load_dotenv
import pymupdf
import os

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Setup RAG
Settings.llm = Anthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# Load documents
print("Loading freight documents...")
documents = SimpleDirectoryReader("documents").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
print("Ready!")

def ask_general(question):
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system="You are a freight and logistics AI assistant for Freight Flex, a 3PL company. Be concise and practical.",
        messages=[{"role": "user", "content": question}]
    )
    return message.content[0].text

def search_documents(question):
    response = query_engine.query(question)
    return str(response)

def read_pdf(pdf_file, question):
    # Open and read the PDF using PyMuPDF
    doc = pymupdf.open(pdf_file.name)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()

    # Send to Claude
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system="You are a freight and logistics AI assistant. Extract and analyze information from the document provided. Be concise and practical.",
        messages=[{
            "role": "user",
            "content": f"Here is the document:\n\n{text}\n\nQuestion: {question}"
        }]
    )
    return message.content[0].text

with gr.Blocks(title="Freight Flex AI Assistant", theme=gr.themes.Base(
    primary_hue="green",
    secondary_hue="green",
    neutral_hue="gray",
)) as app:
    gr.Markdown("# Freight Flex AI Assistant")

    with gr.Tab("Chat"):
        gr.Markdown("### Ask any freight question")
        chat_input = gr.Textbox(placeholder="e.g. What is LTL shipping?")
        chat_output = gr.Textbox(label="Answer")
        chat_btn = gr.Button("Ask", variant="primary")
        chat_btn.click(ask_general, inputs=chat_input, outputs=chat_output)

    with gr.Tab("Search Documents"):
        gr.Markdown("### Ask questions about your saved freight documents")
        rag_input = gr.Textbox(placeholder="e.g. Which invoice has the highest rate?")
        rag_output = gr.Textbox(label="Answer")
        rag_btn = gr.Button("Search", variant="primary")
        rag_btn.click(search_documents, inputs=rag_input, outputs=rag_output)

    with gr.Tab("Upload PDF"):
        gr.Markdown("### Upload any freight PDF and ask questions about it")
        pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        pdf_question = gr.Textbox(placeholder="e.g. What is the total rate? Who is the shipper?")
        pdf_output = gr.Textbox(label="Answer")
        pdf_btn = gr.Button("Analyze PDF", variant="primary")
        pdf_btn.click(read_pdf, inputs=[pdf_input, pdf_question], outputs=pdf_output)

app.launch()