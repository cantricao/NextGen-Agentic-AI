from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.bank.state import BankAgentState
from src.bank.tools.bank_tools import calculate_dti, search_nearest_branch, get_bank_faq
import os

# Initialize the LLM (Using Gemini as configured in your Colab)
# Adjust temperature for determinism in banking
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

def router_node(state: BankAgentState) -> BankAgentState:
    """
    The Router Agent acts as the front desk. It analyzes the user query 
    and decides which specialized agent should handle the request.
    """
    messages = state.get("messages", [])
    if not messages:
        return state
        
    user_query = messages[-1].content
    
    system_prompt = (
        "You are the Bank Router Agent. Analyze the user's request and classify it into exactly ONE of the following categories:\n"
        "1. 'loan_agent': If the user asks about calculating debt, applying for a loan, or DTI.\n"
        "2. 'faq_agent': If the user asks about branches, locations, interest rates, fees, or general bank info.\n"
        "Output ONLY the exact category string (e.g., loan_agent or faq_agent)."
    )
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query)
    ])
    
    route_decision = response.content.strip().lower()
    
    # Fallback mechanism
    if route_decision not in ["loan_agent", "faq_agent"]:
        route_decision = "faq_agent"
        
    return {"next_route": route_decision, "current_agent": "router"}

def loan_agent_node(state: BankAgentState) -> BankAgentState:
    """
    The Loan Agent handles complex calculations and financial advice.
    It has access to the calculate_dti tool.
    """
    messages = state.get("messages", [])
    
    # Bind the specific tool to the LLM
    loan_llm = llm.bind_tools([calculate_dti])
    
    system_prompt = SystemMessage(
        content="You are a strict but helpful Loan Expert Agent. Use the calculate_dti tool if the user provides income and debt. Explain the risk clearly."
    )
    
    # Invoke LLM with history
    response = loan_llm.invoke([system_prompt] + messages)
    
    return {"messages": [response], "current_agent": "loan_agent", "num_steps": state.get("num_steps", 0) + 1}

def faq_agent_node(state: BankAgentState) -> BankAgentState:
    """
    The FAQ Agent handles general inquiries and branch locations.
    It has access to search_nearest_branch and get_bank_faq tools.
    """
    messages = state.get("messages", [])
    location = state.get("user_location", "Unknown")
    
    faq_llm = llm.bind_tools([search_nearest_branch, get_bank_faq])
    
    system_prompt = SystemMessage(
        content=f"You are a friendly Bank Support Agent. The user is currently located in {location}. Use your tools to answer their questions accurately."
    )
    
    response = faq_llm.invoke([system_prompt] + messages)
    
    return {"messages": [response], "current_agent": "faq_agent", "num_steps": state.get("num_steps", 0) + 1}
