from typing import TypedDict, Annotated, List, Sequence
import operator
from langchain_core.messages import BaseMessage

class BankAgentState(TypedDict):
    """
    Represents the central state of the banking multi-agent system.
    """
    # Standard LangGraph message history
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    # Custom state variables for banking context
    current_agent: str
    user_location: str
    customer_no: str
    num_steps: int
    
    # Internal routing decision
    next_route: str
