import os
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv

from src.utils.logger import logger
from src.protocols.shopify_cs import shopify_graph

load_dotenv()

app = FastAPI(
    title="NextGen Agentic AI API Gateway",
    description="Secure, async, production-ready endpoints with Human-in-the-Loop support.",
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
async def health_check():
    """Async liveness probe endpoint."""
    return {"status": "healthy", "service": "NextGen-Agentic-AI"}

@app.post("/v1/agent/invoke", dependencies=[Depends(verify_api_key)], tags=["Core Agents"])
async def invoke_agent(user_message: str, session_id: str):
    """
    Asynchronously invokes the agent. Handles high-traffic loads efficiently.
    """
    logger.info(f"Agent invoked asynchronously by session_id: {session_id}")
    config = {"configurable": {"thread_id": session_id}}
    
    try:
        # Use await and ainvoke() for non-blocking execution
        result = await shopify_graph.ainvoke({"messages": [("user", user_message)]}, config=config)
        
        # Async check of the graph state
        current_state = shopify_graph.get_state(config)
        
        if current_state.next and "tools" in current_state.next:
            logger.info(f"Session {session_id} paused for Human-in-the-Loop approval.")
            return {
                "status": "requires_approval",
                "message": "Agent proposed a high-risk action. Waiting for admin approval.",
                "pending_tools": result["messages"][-1].tool_calls
            }
            
        return {
            "status": "success", 
            "response": result["messages"][-1].content
        }
        
    except Exception as e:
        logger.error(f"Workflow execution failed asynchronously. Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Agent Error")

@app.post("/v1/agent/approve", dependencies=[Depends(verify_api_key)], tags=["Human-in-the-Loop"])
async def approve_action(session_id: str):
    """
    Asynchronous admin endpoint to approve and resume a paused workflow.
    """
    config = {"configurable": {"thread_id": session_id}}
    current_state = shopify_graph.get_state(config)
    
    if not current_state.next or "tools" not in current_state.next:
        raise HTTPException(status_code=400, detail="No pending actions awaiting approval for this session.")
    
    logger.info(f"Admin approved actions for session_id: {session_id}. Resuming asynchronously...")
    
    # Resume the graph non-blocking
    result = await shopify_graph.ainvoke(None, config=config)
    
    return {
        "status": "action_executed",
        "response": result["messages"][-1].content
    }
