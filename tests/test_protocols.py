import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from langchain_core.messages import AIMessage

from src.protocols.shopify_cs import ShopifyTicketState, support_agent_node
from src.protocols.orchestrator import A2AState, a2a_supervisor_node

# ==========================================
# 1. TESTS FOR SHOPIFY CS PROTOCOL (HITL)
# ==========================================

@patch("src.protocols.shopify_cs.llm_with_tools")
def test_shopify_agent_propose_refund_triggers_hitl(mock_llm_with_tools):
    """
    Verify that if the AI decides to propose a refund, 
    the system correctly flags the state to trigger Human-in-the-Loop (HITL).
    """
    mock_tool_call = {
        "name": "propose_refund",
        "args": {"order_id": "ORD-999"},
        "id": "call_refund_mock_123"
    }
    
    # We are using .ainvoke() now, so we must mock it as an Async function
    mock_llm_with_tools.ainvoke = AsyncMock(return_value=AIMessage(content="", tool_calls=[mock_tool_call]))
    
    state = ShopifyTicketState(messages=[], ticket_id="ORD-999", requires_human_approval=False)
    
    # Since support_agent_node is async, we must run it inside an event loop
    new_state = asyncio.run(support_agent_node(state))
    
    assert new_state["requires_human_approval"] is True
    assert len(new_state["messages"]) == 1

@patch("src.protocols.shopify_cs.llm_with_tools")
def test_shopify_agent_status_check_bypasses_hitl(mock_llm_with_tools):
    """
    Verify that normal informational queries do NOT trigger the HITL pause.
    """
    # Mock async .ainvoke()
    mock_llm_with_tools.ainvoke = AsyncMock(return_value=AIMessage(content="Your order is currently in transit."))
    
    state = ShopifyTicketState(messages=[], ticket_id="ORD-999", requires_human_approval=False)
    
    # Await the async node execution
    new_state = asyncio.run(support_agent_node(state))
    
    assert new_state["requires_human_approval"] is False

# ==========================================
# 2. TESTS FOR A2A ORCHESTRATOR PROTOCOL
# ==========================================

@patch("src.protocols.orchestrator.llm")
def test_a2a_supervisor_routing_clinical(mock_llm):
    """
    Verify that the A2A Supervisor correctly delegates medical queries 
    to the clinical_agent based on structured output.
    """
    mock_decision = MagicMock(route="clinical_agent")
    mock_llm.with_structured_output.return_value.invoke.return_value = mock_decision
    
    state = A2AState(messages=[], next_sub_agent="")
    new_state = a2a_supervisor_node(state)
    
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
