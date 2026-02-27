import asyncio
import os
import shutil
from typing import Any
from openai import AsyncOpenAI

# Disable internal tracing for a cleaner console output in production
os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"

# Import high-level orchestration framework modules
from agents import set_default_openai_client, set_default_openai_api
from agents import Agent, Runner
from agents.mcp import MCPServerSse
from agents.model_settings import ModelSettings

# Configure the custom AsyncOpenAI client to route through the Gemini API endpoint
custom_client = AsyncOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/", 
    api_key=os.environ.get('GEMINI_API_KEY', os.environ.get('GOOGLE_API_KEY'))
)
set_default_openai_client(custom_client)
set_default_openai_api("chat_completions")

# Define the core language model
MODEL = "gemini-2.5-flash"

async def run_enterprise_agent(mcp_server: MCPServerSse):
    """
    Initializes the Autonomous Agent using a high-level Runner framework.
    Unlike manual LLM calls, this framework automatically handles the ReAct 
    (Reason + Act) loop, tool execution, and message history appending.
    """
    # Define the Agent Entity with strict instructions and bound MCP server
    agent = Agent(
        name="Senior_Bank_Advisor",
        model=MODEL,
        instructions="You are a strict bank advisor. Use the available MCP tools to answer financial questions accurately.",
        mcp_servers=[mcp_server],
        model_settings=ModelSettings(tool_choice="auto"),
    )

    # Define the banking inquiry
    message = "My monthly income is $15000 and my current debts are $4500. Calculate my DTI and assess my loan risk."
    print(f"\n👤 [User Input]: {message}")
    print("🤖 [Agent Engine]: Starting autonomous orchestration (Thought -> Action -> Observation)...\n")
    
    # The Runner abstracts away the manual while loop. It automatically executes
    # the tool over the network and feeds the result back to the LLM.
    result = await Runner.run(
        starting_agent=agent, 
        input=[{"role": "user", "content": message}], 
        max_turns=10
    )
    
    print("\n✅ [Final Agent Synthesis]:")
    print(result.final_output)

async def main():
    """Entry point to establish the SSE connection and trigger the agent Runner."""
    print("--- [Initializing High-Level Agent Framework via SSE] ---")
    
    # Establish the SSE connection using the framework's native MCP wrapper
    async with MCPServerSse(
        name="Enterprise Bank SSE Server",
        params={
            "url": "http://127.0.0.1:8000/sse",
        },
    ) as server:
        await run_enterprise_agent(server)

if __name__ == "__main__":
    # Execute the asynchronous event loop
    asyncio.run(main())
