# =====================================================================
# MUST BE AT THE ABSOLUTE TOP TO SUPPRESS ADK EXPERIMENTAL WARNINGS
# =====================================================================
import os
import warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.simplefilter("ignore")

import asyncio
import logging
import sys
import threading
import time

import httpx
import nest_asyncio
import uvicorn
from dotenv import load_dotenv

# --- GOOGLE A2A & ADK CORE IMPORTS ---
from a2a.client import ClientConfig, ClientFactory, create_text_message_object
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, TransportProtocol
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH

from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor, A2aAgentExecutorConfig
from google.adk.agents import Agent, LlmAgent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

load_dotenv()
logging.basicConfig(level=logging.ERROR, format='%(message)s')

# =====================================================================
# 🛠️ COMPATIBILITY PATCH 
# =====================================================================
from a2a.client import client as real_client_module
from a2a.client.card_resolver import A2ACardResolver

class PatchedClientModule:
    """Hotfix for missing A2ACardResolver in a2a-sdk."""
    def __init__(self, real_module) -> None:
        for attr in dir(real_module):
            if not attr.startswith('_'):
                setattr(self, attr, getattr(real_module, attr))
        self.A2ACardResolver = A2ACardResolver

sys.modules['a2a.client.client'] = PatchedClientModule(real_client_module)  # type: ignore

# =====================================================================
# 1. ATOMIC TOOLS 
# =====================================================================
def compute_dti(income: float, debt: float) -> str:
    if income <= 0: return "Error: Income must be > 0."
    dti = (debt / income) * 100
    risk = "High Risk" if dti > 43 else "Low Risk"
    return f"Calculated DTI is {dti:.2f}%. Risk status: {risk}."

def search_branch(city: str) -> str:
    return f"The Wall Street branch in {city} is open 9 AM to 5 PM."

# =====================================================================
# 2. A2A MICROSERVICES (Distributed Agent Topology)
# =====================================================================
# FIX: Use Flash for simple tasks, but PRO for the complex Coordinator
WORKER_MODEL = 'gemini-2.5-flash'
COORDINATOR_MODEL = 'gemini-2.5-pro'

# --- 2A. Remote Worker 1: Loan Specialist (Port 10020) ---
loan_agent = Agent(
    model=WORKER_MODEL,
    name='loan_specialist_agent',
    instruction="You are a Loan Specialist. Only answer financial questions. Use the 'compute_dti' tool.",
    tools=[compute_dti],
)
loan_card = AgentCard(
    name='Loan Specialist', description='Calculates financial risk and DTI ratios.',
    url='http://127.0.0.1:10020', version='1.0',
    capabilities=AgentCapabilities(streaming=True),
    default_input_modes=['text/plain'], default_output_modes=['text/plain'],
    preferred_transport=TransportProtocol.jsonrpc,
    skills=[AgentSkill(id='calc_dti', name='Calculate DTI', description='Computes financial risk.', tags=['finance'])],
)
remote_loan_agent = RemoteA2aAgent(
    name='calc_dti_remote', description='Call this agent ONLY for DTI or loan calculations.',
    agent_card=f'http://127.0.0.1:10020{AGENT_CARD_WELL_KNOWN_PATH}',
)

# --- 2B. Remote Worker 2: Support Specialist (Port 10021) ---
support_agent = Agent(
    model=WORKER_MODEL,
    name='support_specialist_agent',
    instruction="You are a Support Guide. Only answer location questions. Use the 'search_branch' tool.",
    tools=[search_branch],
)
support_card = AgentCard(
    name='Support Specialist', description='Finds physical bank branch locations.',
    url='http://127.0.0.1:10021', version='1.0',
    capabilities=AgentCapabilities(streaming=True),
    default_input_modes=['text/plain'], default_output_modes=['text/plain'],
    preferred_transport=TransportProtocol.jsonrpc,
    skills=[AgentSkill(id='find_branch', name='Find Branch', description='Locates branches.', tags=['location'])],
)
remote_support_agent = RemoteA2aAgent(
    name='find_branch_remote', description='Call this agent ONLY to find bank locations or hours.',
    agent_card=f'http://127.0.0.1:10021{AGENT_CARD_WELL_KNOWN_PATH}',
)

# --- 2C. The Coordinator: Bank Manager (Port 10022) ---
coordinator_agent = LlmAgent(
    name='bank_manager_coordinator',
    model=COORDINATOR_MODEL,
    instruction="""
    You are the Bank Manager Orchestrator. The user query contains TWO distinct requests: a financial calculation AND a location search.
    
    CRITICAL: You MUST use BOTH available tools ('calc_dti_remote' AND 'find_branch_remote') to gather all necessary information. 
    Do NOT generate your final response until you have successfully retrieved BOTH the DTI calculation AND the branch location.
    """,
    sub_agents=[remote_loan_agent, remote_support_agent],
)
coordinator_card = AgentCard(
    name='Bank Manager', description='Main entry point. Orchestrates multiple sub-agents.',
    url='http://127.0.0.1:10022', version='1.0',
    capabilities=AgentCapabilities(streaming=True),
    default_input_modes=['text/plain'], default_output_modes=['application/json'],
    preferred_transport=TransportProtocol.jsonrpc,
    skills=[AgentSkill(id='coordinate_request', name='Coordinate', description='Entry point.', tags=['routing'])],
)

# =====================================================================
# 3. FAST SERVER LAUNCHER 
# =====================================================================
def create_a2a_app(agent, agent_card):
    runner = Runner(
        app_name=agent.name, agent=agent,
        artifact_service=InMemoryArtifactService(), session_service=InMemorySessionService(), memory_service=InMemoryMemoryService()
    )
    executor = A2aAgentExecutor(runner=runner, config=A2aAgentExecutorConfig())
    return A2AStarletteApplication(agent_card=agent_card, http_handler=DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore()))

async def start_server(agent, agent_card, port):
    config = uvicorn.Config(create_a2a_app(agent, agent_card).build(), host='127.0.0.1', port=port, log_level='critical', loop='none')
    await uvicorn.Server(config).serve()

def run_all_servers():
    nest_asyncio.apply()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    print("\n🚀 [Infrastructure] Launching Distributed A2A Microservices...")
    tasks = [
        loop.create_task(start_server(loan_agent, loan_card, 10020)),
        loop.create_task(start_server(support_agent, support_card, 10021)),
        loop.create_task(start_server(coordinator_agent, coordinator_card, 10022)),
    ]
    loop.run_until_complete(asyncio.gather(*tasks))

# =====================================================================
# 4. A2A CLIENT 
# =====================================================================
async def test_a2a_architecture():
    await asyncio.sleep(4)
    print("✅ [Status] All Microservices Online (Ports: 10020, 10021, 10022)")
    
    user_query = "Hello Manager! I make 15000 a month and have 4500 in debt. Can you calculate my DTI? Also, where is the nearest branch in New York?"
    print(f"\n👤 [User Query]: {user_query}")
    print("🤖 [Bank Manager]: Decomposing intent and routing to remote sub-agents via A2A Protocol...\n")
    
    async with httpx.AsyncClient(timeout=120.0) as httpx_client:
        card_resp = await httpx_client.get(f'http://127.0.0.1:10022{AGENT_CARD_WELL_KNOWN_PATH}')
        client = ClientFactory(ClientConfig(httpx_client=httpx_client)).create(AgentCard(**card_resp.json()))
        
        responses = [resp async for resp in client.send_message(create_text_message_object(content=user_query))]
        
        print("✅ [Final Synthesized Response]:")
        try:
            print(responses[0][0].artifacts[0].parts[0].root.text)
        except Exception:
            print("Failed to parse response artifact.")

if __name__ == "__main__":
    threading.Thread(target=run_all_servers, daemon=True).start()
    asyncio.run(test_a2a_architecture())
