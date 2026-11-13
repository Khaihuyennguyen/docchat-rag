# FastAPI's official synchronous test client (built on httpx)
from fastapi.testclient import TestClient

# Import our app factory
from docchat.api import create_app


def test_health(docchat):
    # TestClient boots the app in memory without starting a live network port
    client = TestClient(create_app(docchat))
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["chunks"] > 0


def test_ask_returns_answer_and_sources(docchat):
    client = TestClient(create_app(docchat))
    # Post query to /ask
    res = client.post("/ask", json={"question": "How many vacation days?"})
    assert res.status_code == 200
    body = res.json()
    # Verifies FakeLLM answered with pre-recorded string containing '18'
    assert "18" in body["answer"]
    # Verifies the source dictionary contains the required keys
    assert body["sources"][0].keys() == {"source", "page", "text"}


def test_ask_without_api_key_returns_503(docchat, monkeypatch):
    # Strip any ANTHROPIC_API_KEY from environment variables
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Force app to initialize the real AnthropicLLM rather than the test fake
    docchat._llm = None
    res = TestClient(create_app(docchat)).post("/ask", json={"question": "How many vacation days?"})
    # Verify server correctly returns 503 Service Unavailable
    assert res.status_code == 503
    assert "ANTHROPIC_API_KEY" in res.json()["detail"]


def test_empty_question_is_rejected(docchat):
    client = TestClient(create_app(docchat))
    # Empty string violates Pydantic min_length=1; expect automatic 422
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_ingest_rejects_missing_folder(docchat):
    client = TestClient(create_app(docchat))
    # Non-existent folder raises HTTPException 400
    assert client.post("/ingest", json={"folder": "no/such/folder"}).status_code == 400


def test_index_page_is_served(docchat):
    client = TestClient(create_app(docchat))
    # Verify GET / serves the HTML file containing the app title
    assert "DocChat" in client.get("/").text