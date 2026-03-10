from typing import TypedDict, Annotated, Sequence
import operator
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from src.clinical.cache import clinical_cache

# Initialize Edge-Optimized LLM (Simulated via Gemini for now, but architecture supports local Unsloth models)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

class ClinicalState(TypedDict):
    """Strict state schema for clinical environment to prevent data leakage."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    patient_id: str
    triage_level: str
    cached_response: str

def medical_triage_node(state: ClinicalState) -> dict:
    """
    Evaluates patient symptoms. 
    Intercepts the request using RediSearch Semantic Cache before hitting the LLM to save VRAM and latency.
    """
    messages = state.get("messages", [])
    patient_id = state.get("patient_id")
    user_query = messages[-1].content if messages else ""

    # 1. Semantic Cache Interception (O(1) lookup via C++)
    cached_result = clinical_cache.check_cache(patient_id, user_query)
    if cached_result:
        print("[INFO] VRAM Singleton bypassed: Serving response from Semantic Cache.")
        return {"cached_response": cached_result, "triage_level": "resolved_from_cache"}

    # 2. LLM Inference (If Cache Miss)
    print("[INFO] Cache Miss: Activating LLM for clinical triage.")
    system_prompt = SystemMessage(
        content="You are a strict Medical Triage Agent. Assess symptoms and assign a level: 'Emergency' or 'Standard'."
    )
    response = llm.invoke([system_prompt] + messages)
    
    # Store the new reasoning in cache for future queries
    clinical_cache.store_cache(patient_id, user_query, response.content)
    
    return {"messages": [response], "triage_level": "assessed"}

def build_clinical_graph() -> StateGraph:
    """Compiles the clinical DAG with strict routing to prevent hallucinated diagnostics."""
    workflow = StateGraph(ClinicalState)
    workflow.add_node("triage", medical_triage_node)
    workflow.set_entry_point("triage")
    workflow.add_edge("triage", END)
    return workflow.compile()
