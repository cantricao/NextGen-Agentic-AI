from typing import TypedDict, Annotated, Sequence
import operator
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

class ShopifyTicketState(TypedDict):
    """Tracks the lifecycle of an e-commerce customer service ticket."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    ticket_id: str
    requires_human_approval: bool

# Dummy Tools simulating Shopify/Richpanel APIs
def fetch_order_status(order_id: str) -> str:
    """Fetches order tracking info from Shopify."""
    return f"Order {order_id} is delayed in transit."

def propose_refund(order_id: str) -> str:
    """Proposes a refund. This triggers the HITL pause."""
    return f"Refund successfully executed for {order_id}."

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
tools = [fetch_order_status, propose_refund]
llm_with_tools = llm.bind_tools(tools)

def support_agent_node(state: ShopifyTicketState) -> dict:
    """Analyzes the Richpanel ticket and strictly adheres to SOPs."""
    messages = state.get("messages", [])
    system_prompt = SystemMessage(
        content="You are an E-commerce CS Agent. Follow SOP strictly. If an order is delayed, ALWAYS propose a refund using the tool."
    )
    response = llm_with_tools.invoke([system_prompt] + messages)
    
    requires_approval = False
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tool in response.tool_calls:
            if tool['name'] == 'propose_refund':
                requires_approval = True
                print(f"[ALERT] High-risk action '{tool['name']}' detected. Triggering Human-in-the-Loop.")
                
    return {"messages": [response], "requires_human_approval": requires_approval}

# ENTERPRISE FEATURE: Global MemorySaver to persist state across HTTP requests
memory_checkpointer = MemorySaver()

def build_shopify_mvp_graph():
    """
    Compiles the MVP with a Checkpointer for cross-request state persistence.
    """
    workflow = StateGraph(ShopifyTicketState)
    tool_node = ToolNode(tools)
    
    workflow.add_node("agent", support_agent_node)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    
    def route_to_tools(state: ShopifyTicketState) -> str:
        if state.get("requires_human_approval"):
            return "tools"
        return END

    workflow.add_conditional_edges("agent", route_to_tools)
    workflow.add_edge("tools", "agent")
    
    # Compile with the checkpointer to enable true HITL via API
    return workflow.compile(checkpointer=memory_checkpointer, interrupt_before=["tools"])

# Export a singleton instance for the API to use
shopify_graph = build_shopify_mvp_graph()
