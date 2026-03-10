import os
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv

# Import our custom enterprise logger
from src.utils.logger import logger

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="NextGen Agentic AI API Gateway",
    description="Secure, production-ready endpoints for Multi-Agent orchestration.",
    version="1.0.0"
)

# Define Security Scheme: Require X-API-Key in HTTP headers
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def verify_api_key(api_key: str = Depends(api_key_header)) -> str:
    """
    Middleware to validate the incoming API key against the environment secret.
    Rejects unauthorized access with a 401 HTTP status.
    """
    expected_key = os.getenv("API_SECRET_KEY")
    
    if not expected_key:
        logger.critical("API_SECRET_KEY is missing in the environment! System is vulnerable.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server configuration error."
        )
        
    if api_key != expected_key:
        logger.warning(f"Unauthorized access attempt detected. Provided key: {api_key}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key. Access denied."
        )
        
    return api_key

@app.get("/health", tags=["System"])
def health_check():
    """Liveness probe endpoint for Kubernetes/Docker health checks."""
    logger.info("Health check endpoint pinged.")
    return {"status": "healthy", "service": "NextGen-Agentic-AI"}

@app.post("/v1/agent/invoke", dependencies=[Depends(verify_api_key)], tags=["Core Agents"])
def invoke_agent(user_message: str, session_id: str):
    """
    Secure endpoint to trigger the main Agentic workflow.
    Strictly requires a valid X-API-Key in the request header.
    """
    logger.info(f"Agent invoked by session_id: {session_id}. Processing request...")
    
    try:
        # TODO: Import and execute your LangGraph workflow here
        # graph = create_bank_graph()
        # result = graph.invoke({"messages": [user_message], ...})
        
        logger.info(f"Workflow executed successfully for session_id: {session_id}")
        return {
            "status": "success", 
            "data": "Agent execution completed successfully."
        }
        
    except Exception as e:
        logger.error(f"Workflow execution failed for session_id: {session_id}. Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Agent Error"
        )
