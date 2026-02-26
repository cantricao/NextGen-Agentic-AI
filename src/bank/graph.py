from langgraph.graph import StateGraph, END
from src.bank.state import BankAgentState
from src.bank.nodes import router_node, loan_agent_node, faq_agent_node
from langgraph.checkpoint.memory import MemorySaver

def route_after_router(state: BankAgentState) -> str:
    """
    Determines the next node based on the router's exact classification.
    """
    # We fetch the exact key that was set in nodes.py
    next_route = state.get("next_route")
    
    # Since the router outputs the exact node names ("loan_agent" or "faq_agent"),
    # we can route directly to them!
    if next_route in ["loan_agent", "faq_agent"]:
        return next_route
        
    # Safety fallback
    return END

def create_bank_graph():
    """
    Compiles the Multi-Agent Banking workflow with a persistent memory checkpointer.
    """
    workflow = StateGraph(BankAgentState)
    
    # 1. Add Nodes
    workflow.add_node("router", router_node)
    workflow.add_node("loan_agent", loan_agent_node)
    workflow.add_node("faq_agent", faq_agent_node)
    
    # 2. Define Edges & Routing
    workflow.set_entry_point("router")
    
    # Connect the router to the routing logic
    workflow.add_conditional_edges("router", route_after_router)
    
    # Both specialized agents end the conversation after doing their job
    workflow.add_edge("loan_agent", END)
    workflow.add_edge("faq_agent", END)
    
    # 3. Initialize MemorySaver to track conversation history per thread
    memory = MemorySaver()
    
    # Compile with checkpointer
    return workflow.compile(checkpointer=memory)