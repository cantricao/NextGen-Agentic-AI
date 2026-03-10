from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from src.bank.state import BankAgentState
from src.bank.tools.bank_tools import calculate_dti, search_nearest_branch, get_bank_faq

# Initialize the LLM with low temperature for deterministic banking responses
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

class RouteDecision(BaseModel):
    """Schema enforcing strict routing outputs from the LLM."""
    route: str = Field(description="Must be exactly 'loan_agent' or 'faq_agent'")

def router_node(state: BankAgentState) -> dict:
    """
    Analyzes user intent and routes to the appropriate specialized agent.
    Utilizes Structured Output to guarantee format compliance without string manipulation.
    """
    messages = state.get("messages", [])
    if not messages:
        return {"next_route": "faq_agent"} 
        
    system_prompt = (
        "You are the Bank Router Agent. Classify the user's intent.\n"
        "1. If related to debt calculation, applying for loans, or DTI assessment -> 'loan_agent'.\n"
        "2. If related to branch locations, interest rates, or general policies -> 'faq_agent'."
    )
    
    router_llm = llm.with_structured_output(RouteDecision)
    decision = router_llm.invoke([SystemMessage(content=system_prompt)] + messages)
    
    return {"next_route": decision.route}

def faq_agent_node(state: BankAgentState) -> dict:
    """
    Handles general inquiries and location-based searches.
    Binds tools but DOES NOT execute them. LangGraph handles execution.
    """
    messages = state.get("messages", [])
    location = state.get("user_location", "Unknown")
    
    faq_llm = llm.bind_tools([search_nearest_branch, get_bank_faq])
    system_prompt = SystemMessage(
        content=f"You are a Support Agent. The user is located in {location}. Use tools if required."
    )
    
    response = faq_llm.invoke([system_prompt] + messages)
    return {"messages": [response], "current_agent": "faq_agent"}

def loan_agent_node(state: BankAgentState) -> dict:
    """
    Handles financial calculations and risk assessments.
    Strictly enforces DTI calculations before providing financial advice.
    """
    messages = state.get("messages", [])
    
    loan_llm = llm.bind_tools([calculate_dti])
    system_prompt = SystemMessage(
        content="You are a strict Bank Loan Agent. ALWAYS utilize the 'calculate_dti' tool for financial assessments."
    )
    
    response = loan_llm.invoke([system_prompt] + messages)
    return {"messages": [response], "current_agent": "loan_agent"}
