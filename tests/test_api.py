from fastapi.testclient import TestClient

def test_health_check(client: TestClient):
    """Ensure the Liveness Probe endpoint returns 200 OK and correct JSON."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "NextGen-Agentic-AI"}

def test_invoke_agent_unauthorized(client: TestClient, invalid_headers: dict):
    """Ensure the API strictly rejects requests with invalid or missing API Keys."""
    # Test with invalid key
    response = client.post("/v1/agent/invoke?user_message=hello&session_id=123", headers=invalid_headers)
    assert response.status_code == 401
    # Updated to match the exact string returned by main.py
    assert response.json()["detail"] == "Invalid API Key. Access denied."

    # Test with missing key
    response_no_key = client.post("/v1/agent/invoke?user_message=hello&session_id=123")
    assert response_no_key.status_code == 401

def test_invoke_agent_authorized(client: TestClient, valid_headers: dict):
    """
    Ensure the API accepts valid API Keys.
    Note: It might return 500 internally because the Google API Key is mocked,
    but the authentication layer (401) must pass.
    """
    response = client.post("/v1/agent/invoke?user_message=hello&session_id=123", headers=valid_headers)
    assert response.status_code in [200, 500]  # 401 means security failed!
