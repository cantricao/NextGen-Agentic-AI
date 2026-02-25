from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from src.bank.state import BankAgentState
from src.bank.nodes import router_node, loan_agent_node, faq_agent_node
from src.bank.tools.bank_tools import calculate_dti, search_nearest_branch, get_bank_faq

def route_logic(state: BankAgentState) -> str:
    """
    Conditional routing function to direct the flow based on Router's decision.
    """
    return state.get("next_route", "faq_agent")

def should_continue(state: BankAgentState) -> str:
    """
    Determines if the current agent invoked a tool or if it finished generating a response.
    """
    last_message = state["messages"][-1]
    # If the LLM decided to use a tool, route to the 'tools' node
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    # Otherwise, the interaction is complete
    return END

def create_bank_graph():
    """
    Compiles the state machine for the multi-agent system.
    """
    workflow = StateGraph(BankAgentState)
    
    # Define the nodes
    workflow.add_node("router", router_node)
    workflow.add_node("loan_agent", loan_agent_node)
    workflow.add_node("faq_agent", faq_agent_node)
    
    # Create a ToolNode containing all available tools
    tools = [calculate_dti, search_nearest_branch, get_bank_faq]
    tool_node = ToolNode(tools)
    workflow.add_node("tools", tool_node)
    
    # Define the flow
    workflow.set_entry_point("router")
    
    # Route from router to specific agents
    workflow.add_conditional_edges("router", route_logic)
    
    # Route from agents to tools or END
    workflow.add_conditional_edges("loan_agent", should_continue)
    workflow.add_conditional_edges("faq_agent", should_continue)
    
    # Tools route back to the agent that called them
    # For simplicity, we route back to router to let it re-evaluate, 
    # but in advanced setups, you route back to the specific caller.
    workflow.add_edge("tools", "router")
    
    return workflow.compile()
