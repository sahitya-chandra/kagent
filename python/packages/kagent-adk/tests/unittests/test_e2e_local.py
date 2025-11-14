import pytest  
from unittest.mock import patch, MagicMock  
from fastapi import FastAPI  
from fastapi.testclient import TestClient  
from a2a.types import AgentCard  
from google.adk.agents import BaseAgent  
  
from kagent.adk._a2a import KAgentApp  
  
  
# Fixture for a simple test agent  
@pytest.fixture  
def test_agent():  
    """Create a minimal BaseAgent for testing."""  
    class TestAgent(BaseAgent):  
        async def run(self, context):  
            return "test response"  
      
    return TestAgent()  
  
  
@pytest.fixture  
def agent_card():  
    """Create a test AgentCard."""  
    return AgentCard(  
        name="test-agent",  
        description="Test agent for unit tests",  
        version="1.0.0"  
    )  
  
  
@pytest.fixture  
def kagent_app(test_agent, agent_card):  
    """Create a KAgentApp instance for testing."""  
    return KAgentApp(  
        root_agent=test_agent,  
        agent_card=agent_card,  
        kagent_url="http://localhost:8083",  
        app_name="test-agent"  
    )

def test_build_local_returns_fastapi_app(kagent_app):  
    """Test that build_local returns a FastAPI application."""  
    app = kagent_app.build_local()  
      
    assert isinstance(app, FastAPI)  
    assert app is not None  
  
  
def test_build_local_uses_in_memory_services(kagent_app):  
    """Test that build_local uses in-memory services instead of HTTP-based ones."""  
    with patch('kagent.adk._a2a.InMemorySessionService') as mock_session:  
        with patch('kagent.adk._a2a.InMemoryTaskStore') as mock_task:  
            kagent_app.build_local()  
              
            # Verify in-memory services were instantiated  
            mock_session.assert_called_once()  
            mock_task.assert_called_once()  
  
  
def test_build_local_registers_health_routes(kagent_app):  
    """Test that health check and thread dump routes are registered."""  
    app = kagent_app.build_local()  
      
    # Get all route paths  
    routes = [route.path for route in app.routes]  
      
    assert "/health" in routes  
    assert "/thread_dump" in routes  
  
  
def test_build_local_no_http_client(kagent_app):  
    """Test that build_local doesn't create HTTP client or token service."""  
    with patch('kagent.adk._a2a.httpx.AsyncClient') as mock_client:  
        with patch('kagent.adk._a2a.KAgentTokenService') as mock_token:  
            kagent_app.build_local()  
              
            # These should NOT be called in build_local  
            mock_client.assert_not_called()  
            mock_token.assert_not_called()