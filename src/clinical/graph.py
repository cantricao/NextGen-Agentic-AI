import time
import os
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.clinical.state import ClinicalState
from src.clinical.cache import clinical_cache
from src.clinical.vector_db import clinical_db

# Initialize LLM
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
    """Node 2: Retrieves documents from ChromaDB with metadata filtering, then calls LLM."""
    start_time = time.time()
    query = state.get("query")
    patient_id = state.get("patient_id")
    
    # STEP 1: RETRIEVAL WITH METADATA FILTERING
    # Strictly isolate data based on patient_id to prevent context leakage
    retriever = clinical_db.as_retriever(
        search_kwargs={
            "k": 2, 
            "filter": {"patient_id": patient_id}
        }
    )
    retrieved_docs = retriever.invoke(query)
    
    # Compile retrieved context
    context = "\n---\n".join([doc.page_content for doc in retrieved_docs])
    if not context:
        context = "No specific medical records found for this patient in the database."

    # STEP 2: AUGMENTED GENERATION
    system_prompt = SystemMessage(
        content=f"""You are a Clinical AI Assistant. You are analyzing records for Patient ID: {patient_id}.
        
MEDICAL CONTEXT RETRIEVED:
{context}

Based strictly on the context above, answer the user's query. If the answer is not in the context, state clearly that you do not have enough information."""
    )
    
    response = llm.invoke([system_prompt, HumanMessage(content=query)])
    
    # Store response in cache to reduce latency for future identical queries
    clinical_cache.store_cache(patient_id, query, response.content)
    
    latency = time.time() - start_time
    return {"messages": [response], "response_latency": latency}

def cache_routing(state: ClinicalState) -> str:
    """Routes the graph based on cache evaluation."""
    if state.get("cache_hit"):
        return END
    return "generate_response"

def create_clinical_graph():
    """Compiles the LangGraph workflow for the Clinical Agent."""
    workflow = StateGraph(ClinicalState)
    
    workflow.add_node("check_cache", check_cache_node)
    workflow.add_node("generate_response", generate_response_node)
    
    workflow.set_entry_point("check_cache")
    
    workflow.add_conditional_edges("check_cache", cache_routing)
    workflow.add_edge("generate_response", END)
    
    return workflow.compile()
