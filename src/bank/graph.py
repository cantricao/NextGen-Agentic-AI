from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from src.bank.state import BankAgentState
from src.bank.nodes import router_node, loan_agent_node, faq_agent_node
from src.bank.tools.bank_tools import calculate_dti, search_nearest_branch, get_bank_faq

def route_after_router(state: BankAgentState) -> str:
    """Directs traffic to the specific agent based on the router's exact output."""
    next_route = state.get("next_route")
    if next_route in ["loan_agent", "faq_agent"]:
        return next_route
    return "faq_agent"

def route_after_agent(state: BankAgentState) -> str:
    """
    Inspects the agent's response. If tool calls are requested, routes to the ToolNode.
    Otherwise, terminates the workflow gracefully.
    """
    messages = state.get("messages", [])
    last_message = messages[-1]
    
    if hasattr(last_message, 'tool_calls') and len(last_message.tool_calls) > 0:
        return "tools"
    return END

def create_bank_graph() -> StateGraph:
    """
    Assembles and compiles the LangGraph Directed Acyclic Graph (DAG).
    """
    workflow = StateGraph(BankAgentState)
    
    # Register global tools into a dedicated ToolNode
    all_tools = [calculate_dti, search_nearest_branch, get_bank_faq]
    tool_node = ToolNode(all_tools)
    
    # 1. Register all nodes
    workflow.add_node("router", router_node)
    workflow.add_node("loan_agent", loan_agent_node)
    workflow.add_node("faq_agent", faq_agent_node)
    workflow.add_node("tools", tool_node)
    
    # 2. Define topology and entry point
    workflow.set_entry_point("router")
    workflow.add_conditional_edges("router", route_after_router)
    
    # 3. Agents hand off to tools or exit
    workflow.add_conditional_edges("loan_agent", route_after_agent, {"tools": "tools", END: END})
    workflow.add_conditional_edges("faq_agent", route_after_agent, {"tools": "tools", END: END})
    
    # 4. Tools return execution context back to the calling agent
    def route_tool_to_agent(state: BankAgentState) -> str:
        return state.get("current_agent", "faq_agent")
        
    workflow.add_conditional_edges("tools", route_tool_to_agent)
    
    return workflow.compile()
