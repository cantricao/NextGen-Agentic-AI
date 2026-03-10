from langgraph.graph import END
from langchain_core.messages import AIMessage
from src.bank.state import BankAgentState
from src.bank.graph import route_after_router, route_after_agent

def test_route_after_router_deterministic():
    """Verify that the router enforces strict destination paths based on state."""
    
    # Test Loan Agent Route
    state_loan = BankAgentState(next_route="loan_agent")
    assert route_after_router(state_loan) == "loan_agent"

    # Test FAQ Agent Route
    state_faq = BankAgentState(next_route="faq_agent")
    assert route_after_router(state_faq) == "faq_agent"

    # Test Fallback Mechanism for invalid outputs
    state_invalid = BankAgentState(next_route="hallucinated_route")
    assert route_after_router(state_invalid) == "faq_agent" # Must default to safe fallback

def test_route_after_agent_with_tool_calls():
    """Verify that agents requesting tools are routed to the ToolNode."""
    mock_message = AIMessage(content="", tool_calls=[{"name": "calculate_dti"}])
    state = BankAgentState(messages=[mock_message])
    
    assert route_after_agent(state) == "tools"

def test_route_after_agent_without_tool_calls():
    """Verify that agents without tool calls terminate the DAG correctly."""
    mock_message = AIMessage(content="Your DTI ratio is 35%. You are approved.")
    state = BankAgentState(messages=[mock_message])
    
    assert route_after_agent(state) == END
