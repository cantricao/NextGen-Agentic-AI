from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage

class ClinicalState(TypedDict):
    """
    Represents the state for the Clinical RAG Agent.
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    patient_id: str
    query: str
    
    # Track caching performance metrics
    cache_hit: bool
    response_latency: float
