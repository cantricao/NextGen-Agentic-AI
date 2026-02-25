import time
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.clinical.state import ClinicalState
from src.clinical.cache import clinical_cache
import os

# Initialize the LLM (Using Gemini 2.5)
# Ensure GOOGLE_API_KEY is loaded in your environment
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

def check_cache_node(state: ClinicalState) -> ClinicalState:
    """Node 1: Intercepts the query and checks the Semantic Cache."""
    start_time = time.time()
    query = state.get("query")
    patient_id = state.get("patient_id")
    
    cached_response = clinical_cache.check_cache(patient_id, query)
    
    if cached_response:
        msg = AIMessage(content=f"[⚡ CACHE HIT] {cached_response}")
        latency = time.time() - start_time
        return {"messages": [msg], "cache_hit": True, "response_latency": latency}
        
    return {"cache_hit": False}

def generate_response_node(state: ClinicalState) -> ClinicalState:
    """Node 2: Calls LLM if cache miss, then stores the result."""
    start_time = time.time()
    query = state.get("query")
    patient_id = state.get("patient_id")
    messages = state.get("messages", [])
    
    system_prompt = SystemMessage(
        content=f"You are a Clinical AI Assistant. You are analyzing records for Patient ID: {patient_id}. Provide accurate, medical-grade information."
    )
    
    # Use only the system prompt and the latest query for generating response
    response = llm.invoke([system_prompt, HumanMessage(content=query)])
    
    # Store the newly generated response in the cache
    clinical_cache.store_cache(patient_id, query, response.content)
    
    latency = time.time() - start_time
    return {"messages": [response], "response_latency": latency}

def cache_routing(state: ClinicalState) -> str:
    """Routes the graph based on cache hit or miss."""
    if state.get("cache_hit"):
        return END
    return "generate_response"

def create_clinical_graph():
    """Compiles the Clinical RAG architecture."""
    workflow = StateGraph(ClinicalState)
    
    workflow.add_node("check_cache", check_cache_node)
    workflow.add_node("generate_response", generate_response_node)
    
    workflow.set_entry_point("check_cache")
    
    workflow.add_conditional_edges("check_cache", cache_routing)
    workflow.add_edge("generate_response", END)
    
    return workflow.compile()
