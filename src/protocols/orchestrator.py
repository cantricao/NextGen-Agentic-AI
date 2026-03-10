from typing import TypedDict, Annotated, Sequence
import operator
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, SystemMessage
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

class A2AState(TypedDict):
    """Global state bridging multiple specialized autonomous sub-graphs."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_sub_agent: str

class SupervisorDecision(BaseModel):
    route: str = Field(description="Must be 'clinical_agent', 'bank_agent', or 'FINISH'")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

def a2a_supervisor_node(state: A2AState) -> dict:
    """
    The Master Orchestrator. Evaluates the conversation and delegates tasks 
    to specific domain experts (Agent-to-Agent communication).
    """
    messages = state.get("messages", [])
    system_prompt = SystemMessage(
        content=(
            "You are the A2A Supervisor. Review the conversation.\n"
            "Route to 'clinical_agent' for medical queries.\n"
            "Route to 'bank_agent' for financial queries.\n"
            "If the request is fully resolved, output 'FINISH'."
        )
    )
    
    decision = llm.with_structured_output(SupervisorDecision).invoke([system_prompt] + messages)
    print(f"[A2A Orchestrator] Delegating task to: {decision.route}")
    return {"next_sub_agent": decision.route}

# Dummy wrappers for sub-graphs to demonstrate A2A connectivity
def call_clinical_subgraph(state: A2AState) -> dict:
    """Wrapper to invoke the compiled Clinical DAG."""
    print("[A2A] Clinical Agent processing...")
    # clinical_graph.invoke(...)
    return {"messages": [("assistant", "Clinical triage completed.")]}

def call_bank_subgraph(state: A2AState) -> dict:
    """Wrapper to invoke the compiled Bank DAG."""
    print("[A2A] Bank Agent processing...")
    # bank_graph.invoke(...)
    return {"messages": [("assistant", "Financial assessment completed.")]}

def build_a2a_orchestrator() -> StateGraph:
    """Assembles the macroscopic Agent-to-Agent topology."""
    workflow = StateGraph(A2AState)
    
    workflow.add_node("supervisor", a2a_supervisor_node)
    workflow.add_node("clinical_agent", call_clinical_subgraph)
    workflow.add_node("bank_agent", call_bank_subgraph)
    
    workflow.set_entry_point("supervisor")
    
    def a2a_router(state: A2AState) -> str:
        route = state.get("next_sub_agent")
        if route == "FINISH":
            return END
        return route

    workflow.add_conditional_edges("supervisor", a2a_router)
    # Sub-agents always report back to the supervisor
    workflow.add_edge("clinical_agent", "supervisor")
    workflow.add_edge("bank_agent", "supervisor")
    
    return workflow.compile()
