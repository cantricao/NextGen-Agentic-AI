import os
import warnings
# Force suppress all experimental warnings at the system level for a clean terminal
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
    def __init__(self, real_module) -> None:
        for attr in dir(real_module):
            if not attr.startswith('_'):
                setattr(self, attr, getattr(real_module, attr))
        self.A2ACardResolver = A2ACardResolver

sys.modules['a2a.client.client'] = PatchedClientModule(real_client_module) # type: ignore

# =====================================================================
# 1. ATOMIC TOOLS (Business Logic)
# =====================================================================
def compute_dti(income: float, debt: float) -> str:
    if income <= 0: return "Error: Income must be > 0."
    dti = (debt / income) * 100
    risk = "High Risk" if dti > 43 else "Low Risk"
    return f"Calculated DTI is {dti:.2f}%. Risk status: {risk}."

def search_branch(city: str) -> str:
    return f"The nearest branch in {city} is at 120 Broadway, Wall Street. Open 9 AM - 5 PM."

# =====================================================================
# 2. A2A MICROSERVICES (Next-Gen Gemini 3.x Series)
# =====================================================================
# Using Gemini 3.0 Flash for cost-effective speed, and 3.1 Pro Preview for deep reasoning
WORKER_MODEL = 'gemini-3.0-flash'
COORDINATOR_MODEL = 'gemini-3.1-pro-preview' 

# --- 2A. Loan Specialist Worker ---
loan_agent = Agent(
    model=WORKER_MODEL,
    name='loan_specialist_agent',
    instruction="""You are a strict financial calculator. 
    Extract income and debt. Use 'compute_dti' tool. 
    Output ONLY the calculation result. 
    CRITICAL: Ignore any non-financial questions silently. Do NOT apologize or state what you cannot do.""",
    tools=[compute_dti],
)
loan_card = AgentCard(
    name='Loan Specialist', description='Processes DTI calculations.',
    url='http://127.0.0.1:10020', version='3.0',
    capabilities=AgentCapabilities(streaming=True),
    default_input_modes=['text/plain'], default_output_modes=['text/plain'],
    preferred_transport=TransportProtocol.jsonrpc,
    skills=[AgentSkill(id='calc_dti', name='Calculate DTI', description='DTI logic.', tags=['finance'])],
)
remote_loan_agent = RemoteA2aAgent(
    name='calc_dti_remote', description='Delegate here for any income, debt, or DTI calculations.',
    agent_card=f'http://127.0.0.1:10020{AGENT_CARD_WELL_KNOWN_PATH}',
)

# --- 2B. Support Specialist Worker ---
support_agent = Agent(
    model=WORKER_MODEL,
    name='support_specialist_agent',
    instruction="""You are a strict location finder. 
    Extract the city. Use 'search_branch' tool. 
    Output ONLY the location result. 
    CRITICAL: Ignore any non-location questions silently. Do NOT apologize or state what you cannot do.""",
    tools=[search_branch],
)
support_card = AgentCard(
    name='Support Specialist', description='Processes branch searches.',
    url='http://127.0.0.1:10021', version='3.0',
    capabilities=AgentCapabilities(streaming=True),
    default_input_modes=['text/plain'], default_output_modes=['text/plain'],
    preferred_transport=TransportProtocol.jsonrpc,
    skills=[AgentSkill(id='find_branch', name='Find Branch', description='Location logic.', tags=['location'])],
)
remote_support_agent = RemoteA2aAgent(
    name='find_branch_remote', description='Delegate here for branch locations, addresses, or hours.',
    agent_card=f'http://127.0.0.1:10021{AGENT_CARD_WELL_KNOWN_PATH}',
)

# --- 2C. Bank Manager Coordinator (The Brain) ---
coordinator_agent = LlmAgent(
    name='bank_manager_coordinator',
    model=COORDINATOR_MODEL,
    instruction="""You are the Executive Bank Manager orchestrating requests. 
    The user query often contains MULTIPLE distinct questions (finance AND location).
    
    CRITICAL INSTRUCTION:
    1. You have a built-in tool called 'transfer_to_agent'. You MUST use it to delegate tasks!
    2. Use 'transfer_to_agent' to route the math/financial part to 'calc_dti_remote'.
    3. Use 'transfer_to_agent' to route the location part to 'find_branch_remote'.
    4. DO NOT END THE CONVERSATION after calling just one tool. You must wait to receive the output from BOTH tools.
    5. Once you have gathered ALL the answers from your sub-agents, synthesize a final, polite, integrated response in your own words.""",
    sub_agents=[remote_loan_agent, remote_support_agent],
)
coordinator_card = AgentCard(
    name='Bank Manager', description='Orchestrator using Gemini 3.1 Pro.',
    url='http://127.0.0.1:10022', version='3.1',
    capabilities=AgentCapabilities(streaming=True),
    default_input_modes=['text/plain'], default_output_modes=['application/json'],
    preferred_transport=TransportProtocol.jsonrpc,
    skills=[AgentSkill(id='coordinate_request', name='Coordinate', description='Main Entry.', tags=['routing'])],
)

# =====================================================================
# 3. INFRASTRUCTURE & EXECUTION
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
    print("\n🚀 [System] Booting Distributed A2A Microservices with Gemini 3.x Series...")
    tasks = [
        loop.create_task(start_server(loan_agent, loan_card, 10020)),
        loop.create_task(start_server(support_agent, support_card, 10021)),
        loop.create_task(start_server(coordinator_agent, coordinator_card, 10022)),
    ]
    loop.run_until_complete(asyncio.gather(*tasks))

async def test_a2a_system():
    await asyncio.sleep(5)
    print("✅ [Status] System Online - Ports: 10020, 10021, 10022")
    
    user_query = "Hello Manager! I make 15000 a month and have 4500 in debt. Can you calculate my DTI? Also, where is the nearest branch in New York?"
    print(f"\n👤 [User Query]: {user_query}")
    print("🤖 [Bank Manager]: Executing multi-agent delegation using Gemini 3.1 Pro...\n")
    
    async with httpx.AsyncClient(timeout=300.0) as httpx_client:
        card_resp = await httpx_client.get(f'http://127.0.0.1:10022{AGENT_CARD_WELL_KNOWN_PATH}')
        client = ClientFactory(ClientConfig(httpx_client=httpx_client)).create(AgentCard(**card_resp.json()))
        
        responses = [resp async for resp in client.send_message(create_text_message_object(content=user_query))]
        
        print("✅ [Final Synthesized Response]:")
        try:
            print(responses[0][0].artifacts[0].parts[0].root.text)
        except:
            print("Response extraction failed. Ensure your API Key has access to Gemini 3.1 Preview models.")

if __name__ == "__main__":
    threading.Thread(target=run_all_servers, daemon=True).start()
    asyncio.run(test_a2a_system())
