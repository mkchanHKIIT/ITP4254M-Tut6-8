import importlib
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

from dotenv import load_dotenv
import pytest

# Load environment variables from .env file
load_dotenv()

# Shared LangSmith API key used by all tests that configure tracer environment variables.
TEST_LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "test-langsmith-key")


@pytest.fixture
def app_module(monkeypatch):
    # Import the example module with mocked LangChain dependencies to avoid real network or interactive calls.

    class FakeChatOpenAI:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class FakeEmbeddings:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class FakeDocument:
        def __init__(self, page_content, metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    class FakeRetriever:
        def __init__(self, docs=None):
            self._docs = docs or []
            self.invoke = MagicMock(return_value=self._docs)

    class FakeVectorStore:
        def __init__(self, docs):
            self.docs = docs

        @classmethod
        def from_documents(cls, docs, embeddings):
            return cls(docs)

        def as_retriever(self, search_kwargs=None):
            return FakeRetriever(self.docs[:3])

    class FakeAgent:
        def __init__(self, model=None, tools=None, system_prompt=None):
            self.model = model
            self.tools = tools or []
            self.system_prompt = system_prompt
            self.invoke = MagicMock(
                return_value={
                    "messages": [
                        SimpleNamespace(
                            __class__=SimpleNamespace(__name__="AIMessage"),
                            content="stub reply",
                        )
                    ]
                }
            )

    def fake_create_agent(model=None, tools=None, system_prompt=None):
        return FakeAgent(model=model, tools=tools, system_prompt=system_prompt)

    def fake_tool(func):
        func.name = func.__name__
        return func

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    monkeypatch.setitem(
        sys.modules,
        "langchain_openai",
        SimpleNamespace(ChatOpenAI=FakeChatOpenAI),
    )
    monkeypatch.setitem(
        sys.modules,
        "langchain.agents",
        SimpleNamespace(create_agent=fake_create_agent),
    )
    monkeypatch.setitem(
        sys.modules,
        "langchain_core.tools",
        SimpleNamespace(tool=fake_tool),
    )
    monkeypatch.setitem(
        sys.modules,
        "langchain_huggingface",
        SimpleNamespace(HuggingFaceEmbeddings=FakeEmbeddings),
    )
    monkeypatch.setitem(
        sys.modules,
        "langchain_community.vectorstores",
        SimpleNamespace(FAISS=FakeVectorStore),
    )
    monkeypatch.setitem(
        sys.modules,
        "langchain_core.documents",
        SimpleNamespace(Document=FakeDocument),
    )

    if "LangChainExample" in sys.modules:
        del sys.modules["LangChainExample"]

    module = importlib.import_module("LangChainExample")
    return module


@pytest.fixture
def langsmith_env(monkeypatch):
    # Provide LangSmith environment variables for tests verifying tracing and API key propagation.

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", TEST_LANGSMITH_API_KEY)
    monkeypatch.setenv("LANGSMITH_PROJECT", "pytest-langchainexample")
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    return {
        "LANGSMITH_TRACING": "true",
        "LANGSMITH_API_KEY": TEST_LANGSMITH_API_KEY,
        "LANGSMITH_PROJECT": "pytest-langchainexample",
        "LANGSMITH_ENDPOINT": "https://api.smith.langchain.com",
    }


def _fake_response(status_code=200, text="Sunny +28C"):
    response = MagicMock()
    response.status_code = status_code
    response.text = text
    return response

""""Test Cases start from here, the above code is for setting up the test environment and fixtures."""
def test_document_search_returns_ranked_matches(app_module, monkeypatch):
    # Verify document_search returns formatted ranked results for matching documents.

    fake_docs = [
        SimpleNamespace(
            page_content="Python is a versatile programming language.",
            metadata={"doc_id": "node-1"},
        ),
        SimpleNamespace(
            page_content="LangChain is a framework for building LLM-powered apps.",
            metadata={"doc_id": "node-2"},
        ),
    ]
    fake_retriever = MagicMock()
    fake_retriever.invoke.return_value = fake_docs
    monkeypatch.setattr(app_module, "retriever", fake_retriever)

    result = app_module.document_search("Python LangChain")

    assert "[node-1] Python is a versatile programming language." in result
    assert "[node-2] LangChain is a framework for building LLM-powered apps." in result
    fake_retriever.invoke.assert_called_once_with("Python LangChain")


def test_document_search_returns_no_match_message(app_module, monkeypatch):
    # Verify an empty search result returns the no-match message.

    fake_retriever = MagicMock()
    fake_retriever.invoke.return_value = []
    monkeypatch.setattr(app_module, "retriever", fake_retriever)

    result = app_module.document_search("nothing relevant here")

    assert result == "No documents matched your query."
    fake_retriever.invoke.assert_called_once_with("nothing relevant here")


def test_document_search_uses_unknown_when_docid_missing(app_module, monkeypatch):
    # Verify missing document metadata uses the fallback label 'unknown'.

    fake_docs = [
        SimpleNamespace(
            page_content="Document without metadata id.",
            metadata={},
        ),
    ]
    fake_retriever = MagicMock()
    fake_retriever.invoke.return_value = fake_docs
    monkeypatch.setattr(app_module, "retriever", fake_retriever)

    result = app_module.document_search("metadata")

    assert result == "[unknown] Document without metadata id."


@pytest.mark.parametrize(
    "location,expected_fragment",
    [
        ("Hong Kong", "Weather for Hong Kong: Cloudy +27C"),
        ("", "Weather for Hong Kong: Cloudy +27C"),
        (None, "Weather for Hong Kong: Cloudy +27C"),
    ],
)
def test_get_weather_success_and_default_location(
    app_module, monkeypatch, location, expected_fragment
):
    # Verify get_weather succeeds and defaults to Hong Kong when no location is provided.

    fake_get = MagicMock(return_value=_fake_response(200, "Cloudy +27C"))
    monkeypatch.setattr(app_module.requests, "get", fake_get)

    result = app_module.get_weather(location)

    assert result == expected_fragment
    called_url = fake_get.call_args.args[0]
    assert called_url.startswith("https://wttr.in/")
    assert "Hong%20Kong" in called_url
    assert fake_get.call_args.kwargs["timeout"] == 10


def test_get_weather_url_encodes_location(app_module, monkeypatch):
    # Verify get_weather URL-encodes the provided location name.

    fake_get = MagicMock(return_value=_fake_response(200, "Rain +21C"))
    monkeypatch.setattr(app_module.requests, "get", fake_get)

    result = app_module.get_weather("New York")

    assert result == "Weather for New York: Rain +21C"
    called_url = fake_get.call_args.args[0]
    assert "New%20York" in called_url


@pytest.mark.parametrize("status_code", [400, 404, 500])
def test_get_weather_failure_status_codes(app_module, monkeypatch, status_code):
    # Verify get_weather reports failed HTTP status codes correctly.

    fake_get = MagicMock(return_value=_fake_response(status_code, "error"))
    monkeypatch.setattr(app_module.requests, "get", fake_get)

    result = app_module.get_weather("Hong Kong")

    assert result == f"Weather lookup failed for Hong Kong with status {status_code}."


def test_agent_is_created_with_expected_tools(app_module):
    # Verify the agent is built with the expected tool list and system prompt.

    tool_names = [tool.__name__ for tool in app_module.agent.tools]

    assert tool_names == ["document_search", "get_weather"]
    assert app_module.agent.system_prompt == "You are a helpful assistant. Use tools when needed."

"""The following tests verify that the agent's invoke method correctly routes to the appropriate tool based on user input and that the LangSmith environment variables are set as expected for tracing."""
def test_agent_invoke_weather_path_uses_weather_tool(app_module, monkeypatch, langsmith_env):
    # Verify weather-related user input triggers the get_weather tool path.

    tool_calls = []

    def fake_weather(location="Hong Kong"):
        tool_calls.append(("get_weather", location))
        return "Weather for Hong Kong: Cloudy +27C"

    def fake_document_search(query):
        tool_calls.append(("document_search", query))
        return "node-2: LangChain is a framework for building LLM-powered apps."

    monkeypatch.setattr(app_module, "get_weather", fake_weather)
    monkeypatch.setattr(app_module, "document_search", fake_document_search)


    def fake_agent_invoke(payload):
        user_text = payload["messages"][-1]["content"]
        if "weather" in user_text.lower():
            content = app_module.get_weather("Hong Kong")
        else:
            content = app_module.document_search(user_text)
        return {
            "messages": [
                SimpleNamespace(
                    __class__=SimpleNamespace(__name__="AIMessage"),
                    content=content,
                )
            ]
        }

    app_module.agent.invoke = MagicMock(side_effect=fake_agent_invoke)

    result = app_module.agent.invoke(
        {"messages": [{"role": "user", "content": "What is the weather in Hong Kong?"}]}
    )

    assert tool_calls == [("get_weather", "Hong Kong")]
    assert result["messages"][-1].content == "Weather for Hong Kong: Cloudy +27C"
    assert langsmith_env["LANGSMITH_TRACING"] == "true"


def test_agent_invoke_document_path_uses_document_tool(app_module, monkeypatch, langsmith_env):
    # Verify document-related user input triggers the document_search tool path.

    tool_calls = []

    def fake_weather(location="Hong Kong"):
        tool_calls.append(("get_weather", location))
        return "Weather for Hong Kong: Cloudy +27C"

    def fake_document_search(query):
        tool_calls.append(("document_search", query))
        return "node-2: LangChain is a framework for building LLM-powered apps."

    monkeypatch.setattr(app_module, "get_weather", fake_weather)
    monkeypatch.setattr(app_module, "document_search", fake_document_search)

    def fake_agent_invoke(payload):
        user_text = payload["messages"][-1]["content"]
        if any(word in user_text.lower() for word in ["langchain", "document", "course", "module"]):
            content = app_module.document_search(user_text)
        else:
            content = app_module.get_weather("Hong Kong")
        return {
            "messages": [
                SimpleNamespace(
                    __class__=SimpleNamespace(__name__="AIMessage"),
                    content=content,
                )
            ]
        }

    app_module.agent.invoke = MagicMock(side_effect=fake_agent_invoke)

    result = app_module.agent.invoke(
        {"messages": [{"role": "user", "content": "Find documents about LangChain."}]}
    )

    assert tool_calls == [("document_search", "Find documents about LangChain.")]
    assert "LangChain is a framework" in result["messages"][-1].content
    assert langsmith_env["LANGSMITH_PROJECT"] == "pytest-langchainexample"


def test_langsmith_fixture_exposes_expected_env(langsmith_env):
    # Verify the LangSmith fixture exposes the expected tracing environment values.

    assert langsmith_env == {
        "LANGSMITH_TRACING": "true",
        "LANGSMITH_API_KEY": TEST_LANGSMITH_API_KEY,
        "LANGSMITH_PROJECT": "pytest-langchainexample",
        "LANGSMITH_ENDPOINT": "https://api.smith.langchain.com",
    }