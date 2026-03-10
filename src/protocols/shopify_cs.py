import asyncio
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

# ==========================================
# ENTERPRISE ERROR HANDLING (THE "ARMOR")
# ==========================================
def fetch_order_status(order_id: str) -> str:
    """Fetches order tracking info from Shopify with strict error handling."""
    try:
        # Simulate network latency
        # await asyncio.sleep(0.5) # If this were an async tool
        
        # Simulate a random API failure for demonstration
        if order_id.upper() == "FAIL":
            raise TimeoutError("Shopify API is currently unresponsive.")
            
        return f"Order {order_id} is delayed in transit."
    except Exception as e:
        # DO NOT crash the graph. Return the error so the LLM can apologize to the user.
        print(f"[ERROR] fetch_order_status failed: {e}")
        return f"SYSTEM ERROR: Unable to fetch data due to '{str(e)}'. Please politely apologize to the customer and ask them to try again later."

def propose_refund(order_id: str) -> str:
    """Proposes a refund. This triggers the HITL pause."""
    try:
        return f"Refund successfully executed for {order_id}."
    except Exception as e:
        return f"SYSTEM ERROR: Refund failed due to '{str(e)}'."

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
tools = [fetch_order_status, propose_refund]
llm_with_tools = llm.bind_tools(tools)

# ==========================================
# ASYNCHRONOUS NODE EXECUTION
# ==========================================
async def support_agent_node(state: ShopifyTicketState) -> dict:
    """
    Analyzes the Richpanel ticket asynchronously to prevent event-loop blocking.
    """
    messages = state.get("messages", [])
    system_prompt = SystemMessage(
        content="You are an E-commerce CS Agent. Follow SOP strictly. If a tool returns a SYSTEM ERROR, apologize to the user."
    )
    
    # Use ainvoke() instead of invoke() for non-blocking I/O
    response = await llm_with_tools.ainvoke([system_prompt] + messages)
    
    requires_approval = False
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tool in response.tool_calls:
            if tool['name'] == 'propose_refund':
                requires_approval = True
                print(f"[ALERT] High-risk action '{tool['name']}' detected. Triggering Human-in-the-Loop.")
                
    return {"messages": [response], "requires_human_approval": requires_approval}

memory_checkpointer = MemorySaver()

def build_shopify_mvp_graph():
    """Compiles the MVP with Checkpointer and Async support."""
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
    
    return workflow.compile(checkpointer=memory_checkpointer, interrupt_before=["tools"])

shopify_graph = build_shopify_mvp_graph()
