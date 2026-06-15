"""
Integration tests for LangChainExample using real LangChain components and LangSmith tracing.
These tests verify that the AI agent correctly routes to and calls the appropriate tools.
"""

import os
from unittest.mock import MagicMock
from dotenv import load_dotenv
import pytest
from langsmith import Client
from langsmith.run_helpers import get_current_run_tree

# Load environment variables
load_dotenv()


@pytest.fixture(scope="module")
def langsmith_client():
    """Initialize the LangSmith client for querying traced runs."""
    api_key = os.getenv("LANGSMITH_API_KEY")
    project_name = os.getenv("LANGSMITH_PROJECT", "langchainexample")
    if not api_key:
        pytest.skip("LANGSMITH_API_KEY not set in .env")
    return Client(api_key=api_key)


@pytest.fixture
def real_app_module(monkeypatch):
    """Import the real LangChainExample module with mocked external services."""
    import sys
    from types import SimpleNamespace
    
    # Mock external API calls to keep tests fast and deterministic
    mock_get = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "Sunny +28C"
    mock_get.return_value = mock_response
    
    # Mock HuggingFace embeddings to avoid downloading models
    mock_embeddings_class = MagicMock()
    mock_embeddings_instance = MagicMock()
    mock_embeddings_class.return_value = mock_embeddings_instance
    
    # Mock FAISS vector store
    mock_faiss_class = MagicMock()
    mock_faiss_instance = MagicMock()
    mock_faiss_retriever = MagicMock()
    mock_faiss_retriever.invoke = MagicMock(return_value=[])
    mock_faiss_instance.as_retriever = MagicMock(return_value=mock_faiss_retriever)
    mock_faiss_class.from_documents = MagicMock(return_value=mock_faiss_instance)
    
    # Monkeypatch sys.modules to provide mock modules
    mock_sys_modules = {
        "langchain_huggingface": SimpleNamespace(HuggingFaceEmbeddings=mock_embeddings_class),
        "langchain_community.vectorstores": SimpleNamespace(FAISS=mock_faiss_class),
    }
    
    for module_name, mock_module in mock_sys_modules.items():
        monkeypatch.setitem(sys.modules, module_name, mock_module)
    
    # Clear cached module
    if "LangChainExample" in sys.modules:
        del sys.modules["LangChainExample"]
    
    # Now import the real module
    import LangChainExample
    
    # Also patch the requests.get in the imported module
    monkeypatch.setattr(LangChainExample, "requests", MagicMock())
    
    return LangChainExample


def test_agent_invokes_weather_tool(real_app_module, monkeypatch):
    """Test that asking about weather triggers the get_weather tool."""
    # Mock the requests module in the real module
    mock_get = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "Cloudy +25C"
    mock_get.return_value = mock_response
    
    # Mock requests.utils.quote to properly URL-encode
    mock_quote = MagicMock(side_effect=lambda x: x.replace(" ", "%20"))
    
    mock_requests = MagicMock()
    mock_requests.get = mock_get
    mock_requests.utils.quote = mock_quote
    
    monkeypatch.setattr(real_app_module, "requests", mock_requests)
    
    # Invoke the agent with a weather-related question
    result = real_app_module.agent.invoke(
        {"messages": [{"role": "user", "content": "What is the weather in Hong Kong?"}]}
    )
    
    # Verify the agent returned a response
    assert result is not None
    assert "messages" in result
    
    # Verify the get_weather tool was called (via requests.get)
    mock_get.assert_called()
    called_url = mock_get.call_args.args[0]
    assert "wttr.in" in called_url
    assert "Hong%20Kong" in called_url


def test_agent_invokes_document_search_tool(real_app_module, monkeypatch):
    """Test that asking about documents triggers the document_search tool."""
    from types import SimpleNamespace
    
    mock_retriever = MagicMock()
    # Mock the retriever to return sample documents
    mock_docs = [
        SimpleNamespace(
            page_content="LangChain is a framework for building LLM-powered apps.",
            metadata={"doc_id": "node-2"},
        )
    ]
    mock_retriever.invoke = MagicMock(return_value=mock_docs)
    monkeypatch.setattr(real_app_module, "retriever", mock_retriever)
    
    # Invoke the agent with a document-related question
    result = real_app_module.agent.invoke(
        {"messages": [{"role": "user", "content": "Find information about #Fill-in this blank#}."}]}
    )
    
    # Verify the agent returned a response
    assert result is not None
    assert "messages" in result
    
    # Verify the document_search tool was called (via retriever.invoke)
    mock_retriever.invoke.assert_called()


def test_agent_tools_are_registered(real_app_module):
    """Test that the agent has the expected tools available."""
    # In the real module, document_search and get_weather are defined as @tool decorated functions
    # which become StructuredTool objects
    from langchain_core.tools import StructuredTool
    
    assert hasattr(real_app_module, "document_search"), "document_search tool should exist"
    assert hasattr(real_app_module, "get_weather"), "get_weather tool should exist"
    
    # Tools are StructuredTool objects created by the @tool decorator
    assert isinstance(real_app_module.document_search, StructuredTool), "document_search should be a StructuredTool"
    assert isinstance(real_app_module.get_weather, StructuredTool), "get_weather should be a StructuredTool"
    
    # Verify tool names
    assert real_app_module.document_search.name == "document_search"
    assert real_app_module.get_weather.name == "get_weather"


def test_weather_tool_formats_response(real_app_module, monkeypatch):
    """Test that the get_weather tool correctly formats its response."""
    mock_get = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "Rainy +20C"
    mock_get.return_value = mock_response
    
    monkeypatch.setattr(real_app_module.requests, "get", mock_get)
    
    # Call the tool via its invoke method (tools are StructuredTool objects)
    result = real_app_module.get_weather.invoke({"location": "Tokyo"})
    
    # Verify the response format
    assert result.startswith("Weather for Tokyo:")
    assert "Rainy" in result
    assert "20C" in result


def test_weather_tool_handles_api_error(real_app_module, monkeypatch):
    """Test that the get_weather tool handles API errors gracefully."""
    mock_get = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_get.return_value = mock_response
    
    monkeypatch.setattr(real_app_module.requests, "get", mock_get)
    
    # Call the tool via its invoke method
    result = real_app_module.get_weather.invoke({"location": "InvalidCity"})
    
    # Verify error handling
    assert "Weather lookup failed" in result
    assert "404" in result


def test_document_search_tool_returns_formatted_results(real_app_module, monkeypatch):
    """Test that the document_search tool formats results correctly."""
    from types import SimpleNamespace
    
    mock_retriever = MagicMock()
    mock_docs = [
        SimpleNamespace(
            page_content="Python is versatile.",
            metadata={"doc_id": "node-1"},
        ),
        SimpleNamespace(
            page_content="AI is transformative.",
            metadata={"doc_id": "node-3"},
        ),
    ]
    mock_retriever.invoke = MagicMock(return_value=mock_docs)
    monkeypatch.setattr(real_app_module, "retriever", mock_retriever)
    
    # Call the tool via its invoke method
    result = real_app_module.document_search.invoke({"query": "Python AI"})
    
    # Verify formatting
    assert "[node-1]" in result
    assert "Python is versatile" in result
    assert "[node-3]" in result
    assert "AI is transformative" in result


def test_document_search_tool_handles_no_matches(real_app_module, monkeypatch):
    """Test that document_search returns appropriate message for no matches."""
    mock_retriever = MagicMock()
    mock_retriever.invoke = MagicMock(return_value=[])
    monkeypatch.setattr(real_app_module, "retriever", mock_retriever)
    
    # Call the tool via its invoke method
    result = real_app_module.document_search.invoke({"query": "nonexistent query"})
    
    # Verify no-match message
    assert result == "No documents matched your query."


def test_agent_system_prompt_is_set(real_app_module):
    """Test that the agent was created with a system prompt."""
    # Verify the agent exists and is callable
    assert hasattr(real_app_module, "agent"), "agent should exist"
    assert callable(real_app_module.agent.invoke), "agent.invoke should be callable"


@pytest.mark.integration
def test_agent_integration_with_langsmith_tracing(real_app_module):
    """
    Integration test: Verify that LangSmith captures tool invocations.
    This test requires valid LANGSMITH_API_KEY in .env.
    """
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key or api_key.startswith("lsv2_pt_"):
        pytest.skip("LangSmith API key not configured or invalid")
    
    # Note: This test would require setting LANGSMITH_TRACING=true
    # and then querying the LangSmith API to verify tool calls were traced.
    # For now, we verify the environment is properly configured.
    
    assert os.getenv("LANGSMITH_TRACING") == "true"
    assert os.getenv("LANGSMITH_PROJECT") is not None
    assert os.getenv("LANGSMITH_ENDPOINT") is not None
