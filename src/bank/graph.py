from langgraph.graph import StateGraph, END
from src.bank.state import BankAgentState
from src.bank.nodes import router_node, loan_agent_node, faq_agent_node
from langgraph.checkpoint.memory import MemorySaver

def route_after_router(state: BankAgentState) -> str:
    """
    Determines the next node based on the router's intent classification.
    """
    intent = state.get("current_intent")
    if intent == "LOAN_INQUIRY":
        return "loan_agent"
    elif intent == "FAQ_OR_LOCATION":
        return "faq_agent"
    else:
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
    workflow.add_conditional_edges("router", route_after_router)
    
    workflow.add_edge("loan_agent", END)
    workflow.add_edge("faq_agent", END)
    
    # 3. Initialize MemorySaver to track conversation history per thread
    memory = MemorySaver()
    
    # Compile with checkpointer
    return workflow.compile(checkpointer=memory)
