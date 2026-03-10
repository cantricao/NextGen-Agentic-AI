import os
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv

from src.utils.logger import logger
from src.protocols.shopify_cs import shopify_graph

load_dotenv()

app = FastAPI(
    title="NextGen Agentic AI API Gateway",
    description="Secure, production-ready endpoints with Human-in-the-Loop support.",
    version="1.0.0"
)

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def verify_api_key(api_key: str = Depends(api_key_header)) -> str:
    """Middleware to validate the incoming API key against the environment secret."""
    expected_key = os.getenv("API_SECRET_KEY", "super_secret_test_key")
    if api_key != expected_key:
        logger.warning(f"Unauthorized access attempt. Key used: {api_key}")
        raise HTTPException(status_code=401, detail="Invalid API Key. Access denied.")
    return api_key

@app.get("/health", tags=["System"])
def health_check():
    """Liveness probe endpoint."""
    return {"status": "healthy", "service": "NextGen-Agentic-AI"}

@app.post("/v1/agent/invoke", dependencies=[Depends(verify_api_key)], tags=["Core Agents"])
def invoke_agent(user_message: str, session_id: str):
    """
    Invokes the agent. If a high-risk action is detected, it pauses execution
    and returns a 'requires_approval' status.
    """
    logger.info(f"Agent invoked by session_id: {session_id}")
    
    # Configure the thread_id to persist state for this specific user session
    config = {"configurable": {"thread_id": session_id}}
    
    try:
        # Run the graph until it finishes OR hits an interrupt_before breakpoint
        result = shopify_graph.invoke({"messages": [("user", user_message)]}, config=config)
        
        # Check current state of the graph
        current_state = shopify_graph.get_state(config)
        
        # If 'next' contains 'tools', it means the graph is paused waiting for human approval
        if current_state.next and "tools" in current_state.next:
            logger.info(f"Session {session_id} paused for Human-in-the-Loop approval.")
            return {
                "status": "requires_approval",
                "message": "Agent proposed a high-risk action (e.g., refund). Waiting for admin approval.",
                "pending_tools": result["messages"][-1].tool_calls
            }
            
        return {
            "status": "success", 
            "response": result["messages"][-1].content
        }
        
    except Exception as e:
        logger.error(f"Workflow execution failed. Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Agent Error")

@app.post("/v1/agent/approve", dependencies=[Depends(verify_api_key)], tags=["Human-in-the-Loop"])
def approve_action(session_id: str):
    """
    Admin endpoint to approve and resume a paused workflow.
    """
    config = {"configurable": {"thread_id": session_id}}
    current_state = shopify_graph.get_state(config)
    
    if not current_state.next or "tools" not in current_state.next:
        raise HTTPException(status_code=400, detail="No pending actions awaiting approval for this session.")
    
    logger.info(f"Admin approved pending actions for session_id: {session_id}. Resuming workflow...")
    
    # Resume the graph by invoking it with None (continues from checkpoint)
    result = shopify_graph.invoke(None, config=config)
    
    return {
        "status": "action_executed",
        "response": result["messages"][-1].content
    }
