import os
import pytest
from fastapi.testclient import TestClient

# Mock environment variables before importing the application
os.environ["API_SECRET_KEY"] = "super_secret_test_key"
os.environ["GOOGLE_API_KEY"] = "mock_gemini_key_for_testing"

from src.api.main import app

@pytest.fixture
def client() -> TestClient:
    """Provides a global FastAPI TestClient for endpoint testing."""
    return TestClient(app)

@pytest.fixture
def valid_headers() -> dict:
    """Provides valid HTTP headers for authenticated requests."""
    return {"X-API-Key": "super_secret_test_key"}

@pytest.fixture
def invalid_headers() -> dict:
    """Provides invalid HTTP headers to test security rejections."""
    return {"X-API-Key": "wrong_key_123"}
