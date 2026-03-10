import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage

from src.protocols.shopify_cs import ShopifyTicketState, support_agent_node
from src.protocols.orchestrator import A2AState, a2a_supervisor_node

# ==========================================
# 1. TESTS FOR SHOPIFY CS PROTOCOL (HITL)
# ==========================================

# Mock the entire 'llm_with_tools' object instead of just its '.invoke' method
@patch("src.protocols.shopify_cs.llm_with_tools")
def test_shopify_agent_propose_refund_triggers_hitl(mock_llm_with_tools):
    """
    Verify that if the AI decides to propose a refund, 
    the system correctly flags the state to trigger Human-in-the-Loop (HITL).
    """
    # Define the mock tool call based on LangChain's strict schema
    mock_tool_call = {
        "name": "propose_refund",
        "args": {"order_id": "ORD-999"},
        "id": "call_refund_mock_123"
    }
    
    # Configure the mock object's invoke method to return our simulated AIMessage
    mock_llm_with_tools.invoke.return_value = AIMessage(content="", tool_calls=[mock_tool_call])
    
    # Initialize state
    state = ShopifyTicketState(messages=[], ticket_id="ORD-999", requires_human_approval=False)
    
    # Run the agent node
    new_state = support_agent_node(state)
    
    # Assertions: System MUST flag for human approval
    assert new_state["requires_human_approval"] is True
    assert len(new_state["messages"]) == 1

@patch("src.protocols.shopify_cs.llm_with_tools")
def test_shopify_agent_status_check_bypasses_hitl(mock_llm_with_tools):
    """
    Verify that normal informational queries do NOT trigger the HITL pause.
    """
    # Mock LLM returning plain text without any tool calls
    mock_llm_with_tools.invoke.return_value = AIMessage(content="Your order is currently in transit.")
    
    state = ShopifyTicketState(messages=[], ticket_id="ORD-999", requires_human_approval=False)
    new_state = support_agent_node(state)
    
    # Assertions: System must NOT flag for human approval
    assert new_state["requires_human_approval"] is False

# ==========================================
# 2. TESTS FOR A2A ORCHESTRATOR PROTOCOL
# ==========================================

# Mock the entire 'llm' object instead of its dynamic method
@patch("src.protocols.orchestrator.llm")
def test_a2a_supervisor_routing_clinical(mock_llm):
    """
    Verify that the A2A Supervisor correctly delegates medical queries 
    to the clinical_agent based on structured output.
    """
    # Setup the mock decision object
    mock_decision = MagicMock(route="clinical_agent")
    
    # Simulate the chaining: llm.with_structured_output(...).invoke(...)
    mock_llm.with_structured_output.return_value.invoke.return_value = mock_decision
    
    state = A2AState(messages=[], next_sub_agent="")
    new_state = a2a_supervisor_node(state)
    
    # Assertions: Supervisor MUST route to the clinical domain
    assert new_state["next_sub_agent"] == "clinical_agent"

@patch("src.protocols.orchestrator.llm")
def test_a2a_supervisor_routing_finish(mock_llm):
    """
    Verify that the A2A Supervisor gracefully terminates the workflow 
    when the user query is fully resolved.
    """
    mock_decision = MagicMock(route="FINISH")
    mock_llm.with_structured_output.return_value.invoke.return_value = mock_decision
    
    state = A2AState(messages=[], next_sub_agent="")
    new_state = a2a_supervisor_node(state)
    
    assert new_state["next_sub_agent"] == "FINISH"
