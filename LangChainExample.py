import os
from dotenv import load_dotenv
import requests  # For making HTTP requests (e.g., weather API)
from langchain_openai import ChatOpenAI  # For using OpenAI-compatible LLMs
from langchain.agents import create_agent  # For creating an agent that can use tools
from langchain_core.tools import tool  # For defining custom tools
from langchain_huggingface import HuggingFaceEmbeddings  # For generating embeddings using HuggingFace models
from langchain_community.vectorstores import FAISS  # For storing and searching embeddings
from langchain_core.documents import Document  # For representing text documents
# pip install requests langchain-openai langchain-core langchain-huggingface langchain-community faiss-cpu python-dotenv
# pip install langchain

# Load environment variables from .env file
load_dotenv()

# === API and App Configuration ===
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "YOUR-OPENROUTER-API-KEY")  # From .env or fallback
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-nano-30b-a3b:free")  # From .env or fallback
APP_URL = os.getenv("APP_URL", "http://localhost:8000")  # From .env or fallback
APP_TITLE = os.getenv("APP_TITLE", "HK Agent Demo")  # From .env or fallback

# === Initialize the Language Model (LLM) ===
llm = ChatOpenAI(
    model=OPENROUTER_MODEL,
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0,  # Deterministic output
    default_headers={
        "HTTP-Referer": APP_URL,
        "X-OpenRouter-Title": APP_TITLE,
    },
)

# === Prepare Example Documents ===
nodes = [
    Document(
        page_content="Python is a versatile programming language.",
        metadata={"doc_id": "node-1"},
    ),
    Document(
        page_content="LangChain is a framework for building LLM-powered apps.",
        metadata={"doc_id": "node-2"},
    ),
    Document(
        page_content="LlamaIndex helps connect LLMs to your data.",
        metadata={"doc_id": "node-3"},
    ),
    Document(
        page_content="ITP4254M is a module for HD in Applied AI, Robert Chan teaching and training the studnet to be AI applying expert.",
        metadata={"doc_id": "node-4"},
    ),
    Document(
        page_content="Robert Chan is a lecturer in HKIIT, he has been teaching for ard 4 years, he taught students in AI & Smart Technology, Applied AI and Cybersecurity.",
        metadata={"doc_id": "node-5"},
    ),
]

# === Create Embeddings and Vector Store ===
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")  # Use a small English embedding model
vectorstore = FAISS.from_documents(nodes, embeddings)  # Store document embeddings in FAISS for fast search
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Set up retriever to return top 3 matches

# === Tool: Document Search ===
@tool
def document_search(query: str) -> str:
    """Search local course and technical documents."""
    docs = retriever.invoke(query)  # Search for relevant documents
    if not docs:
        return "No documents matched your query."
    # Format and return the matched documents
    return "\n\n".join(
        f"[{d.metadata.get('doc_id', 'unknown')}] {d.page_content}"
        for d in docs
    )

# === Tool: Get Weather ===
@tool
def get_weather(location: str = "Hong Kong") -> str:
    """Get current weather for a city."""
    loc = (location or "Hong Kong").strip()  # Default to Hong Kong if not provided
    # Call wttr.in for weather info
    r = requests.get(f"https://wttr.in/{requests.utils.quote(loc)}?format=%C+%t+%m", timeout=10)
    if r.status_code == 200:
        return f"Weather for {loc}: {r.text.strip()}"
    return f"Weather lookup failed for {loc} with status {r.status_code}."

# === Create the Agent with Tools ===
agent = create_agent(
    model=llm,
    tools=[document_search, get_weather],  # Register the tools, When added tools remember to add tools here
    system_prompt="You are a helpful assistant. Use tools when needed.",
)

# === Main Loop: Chat with the Agent ===
def main():
    while True:
        user_input = input("User: ").strip()  # Get user input
        if user_input.lower() in {"exit", "quit"}:
            break  # Exit loop if user types exit/quit
        result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})  # Send input to agent
        # Print the AI's response
        for msg in reversed(result["messages"]):
            if msg.__class__.__name__ == "AIMessage":
                print(msg.content)
                break


if __name__ == "__main__":
    main()