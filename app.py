import anthropic
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.anthropic import Anthropic
from llama_index.core import Settings
import os

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Setup RAG
Settings.llm = Anthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

def ask_claude(question, document=None):
    if document:
        content = f"Here is a freight document:\n\n{document}\n\nUser question: {question}"
    else:
        content = question

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system="You are a freight and logistics AI assistant for Freight Flex, a 3PL company. Help users with shipping rates, load management, carrier selection, document processing, and freight operations. When given a document, extract key information like shipper, receiver, weight, rate, and delivery details. Be concise and practical.",
        messages=[
            {"role": "user", "content": content}
        ]
    )
    return message.content[0].text

def start_rag():
    print("\nLoading freight documents...")
    documents = SimpleDirectoryReader("documents").load_data()
    print(f"Loaded {len(documents)} documents!")
    print("Building search index...")
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    print("Ready! Ask questions about your freight documents.\n")
    while True:
        question = input("You: ")
        if question.lower() == "quit":
            break
        response = query_engine.query(question)
        print(f"\nAssistant: {response}\n")

print("=== Freight Flex AI Assistant ===")
print("1. Chat mode - ask any freight question")
print("2. Document mode - paste a freight document and extract info")
print("3. RAG mode - ask questions about your saved freight documents")
print("Type 'quit' to exit\n")

mode = input("Choose mode (1, 2 or 3): ")

if mode == "2":
    print("\nPaste your freight document below, then press Enter twice:\n")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
        document = "\n".join(lines)
    question = input("\nWhat do you want to know about this document? ")
    while question != "quit":
        response = ask_claude(question, document)
        print(f"\nAssistant: {response}\n")
        question = input("\nWhat do you want to know about this document? ")

elif mode == "3":
    start_rag()

else:
    while True:
        question = input("You: ")
        if question.lower() == "quit":
            break
        response = ask_claude(question)
        print(f"\nAssistant: {response}\n")