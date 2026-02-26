from langchain_core.messages import ToolMessage
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
        "2. 'faq_agent': If the user asks about branches, locations, specific cities (e.g., 'Chicago', 'nearest branch'), ATMs, interest rates, fees, or general bank info.\n"
        "Output ONLY the exact category string :loan_agent or faq_agent."
    )
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query)
    ])
    
    # Bulletproof cleaning: aggressively strip all quotes, backticks, spaces, and newlines
    route_decision = response.content.lower()
    for char in ["`", '"', "'", "\n", " "]:
        route_decision = route_decision.replace(char, "")    
    
    # Fallback mechanism
    if route_decision not in ["loan_agent", "faq_agent"]:
        print(f"[WARNING] Invalid route parsed: '{route_decision}'. Defaulting to faq_agent.")
        route_decision = "faq_agent"
    
    print(f"🔄 Routed via: {route_decision}")
        
    return {"next_route": route_decision, "current_agent": "router"}

def loan_agent_node(state: BankAgentState) -> BankAgentState:
    """
    The Loan Agent handles complex calculations and financial advice.
    It now autonomously executes tools and formulates a final response.
    """
    messages = state.get("messages", [])
    loan_llm = llm.bind_tools([calculate_dti])
    
    system_prompt = SystemMessage(
        content="You are a strict but helpful Loan Expert Agent. Use the calculate_dti tool if the user provides income and debt. Explain the risk clearly."
    )
    
    # 1. Ask the LLM to decide what to do
    response = loan_llm.invoke([system_prompt] + messages)
    new_messages = [response]
    
    # 2. If the LLM decides to use a tool, execute it automatically!
    if response.tool_calls:
        for tool_call in response.tool_calls:
            # Execute the correct tool
            if tool_call["name"] == "calculate_dti":
                tool_result = calculate_dti.invoke(tool_call["args"])
            else:
                tool_result = "Tool not recognized."
            
            # Pack the result into a ToolMessage
            tool_msg = ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
            new_messages.append(tool_msg)
        
        # 3. Feed the tool's result back to the LLM so it can talk to the user
        final_response = loan_llm.invoke([system_prompt] + messages + new_messages)
        new_messages.append(final_response)
        
    return {"messages": new_messages, "current_agent": "loan_agent", "next_route": "loan_agent"}


def faq_agent_node(state: BankAgentState) -> BankAgentState:
    """
    The FAQ Agent handles general inquiries and branch locations.
    It now autonomously executes tools and formulates a final response.
    """
    messages = state.get("messages", [])
    location = state.get("user_location", "Unknown")
    
    faq_llm = llm.bind_tools([search_nearest_branch, get_bank_faq])
    system_prompt = SystemMessage(
        content=f"You are a friendly Bank Support Agent. The user is currently located in {location}. Use your tools to answer their questions accurately."
    )
    
    # 1. Ask the LLM to decide what to do
    response = faq_llm.invoke([system_prompt] + messages)
    new_messages = [response]
    
    # 2. If the LLM decides to use a tool, execute it automatically!
    if response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "search_nearest_branch":
                tool_result = search_nearest_branch.invoke(tool_call["args"])
            elif tool_call["name"] == "get_bank_faq":
                tool_result = get_bank_faq.invoke(tool_call["args"])
            else:
                tool_result = "Tool not recognized."
                
            tool_msg = ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
            new_messages.append(tool_msg)
            
        # 3. Feed the tool's result back to the LLM to generate the final text
        final_response = faq_llm.invoke([system_prompt] + messages + new_messages)
        new_messages.append(final_response)

    return {"messages": new_messages, "current_agent": "faq_agent", "next_route": "faq_agent"}