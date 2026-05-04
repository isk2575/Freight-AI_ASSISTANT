import gradio as gr
import anthropic
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.anthropic import Anthropic
from llama_index.core import Settings
from dotenv import load_dotenv
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
#read doc and build idx
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
print("Ready!")

def chat(question):
    response = query_engine.query(question)
    return str(response)

def ask_general(question):
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system="You are a freight and logistics AI assistant for Freight Flex, a 3PL company. Be concise and practical.",
        messages=[{"role": "user", "content": question}]
    )
    return message.content[0].text

with gr.Blocks(title="Freight Flex AI Assistant") as app:
    gr.Markdown("# 🚚 Freight Flex AI Assistant")
    
    with gr.Tab("Chat"):
        gr.Markdown("### Ask any freight question")
        chat_input = gr.Textbox(placeholder="e.g. What is LTL shipping?")
        chat_output = gr.Textbox(label="Answer")
        chat_btn = gr.Button("Ask", variant="primary")
        chat_btn.click(ask_general, inputs=chat_input, outputs=chat_output)

    with gr.Tab("Search Documents"):
        gr.Markdown("### Ask questions about your freight documents")
        rag_input = gr.Textbox(placeholder="e.g. Which invoice has the highest rate?")
        rag_output = gr.Textbox(label="Answer")
        rag_btn = gr.Button("Search", variant="primary")
        rag_btn.click(chat, inputs=rag_input, outputs=rag_output)

app.launch()