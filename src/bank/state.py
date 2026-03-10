from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage

class BankAgentState(TypedDict):
    """
    Represents the centralized state for the Bank Multi-Agent System.
    Tracks conversation history, context, and routing directives.
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    # Context variables
    user_location: str
    customer_no: str
    
    # Orchestration variables
    current_agent: str
    next_route: str
