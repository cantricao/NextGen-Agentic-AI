import asyncio
import json
import os
from contextlib import AsyncExitStack
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.sse import sse_client
from openai import AsyncOpenAI

# Load environment variables (Ensure GEMINI_API_KEY is available)
load_dotenv()

# Initialize the AsyncOpenAI client pointing to Gemini's compatibility endpoint
client = AsyncOpenAI(
    api_key=os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY")), 
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
MODEL = "gemini-2.5-flash"

async def run_interactive_agent():
    """
    Production-grade Interactive Chat Client.
    Connects to the MCP Server over SSE, loads available tools dynamically, 
    and enters a continuous Chat Loop for real-time user interaction.
    """
    server_url = "http://127.0.0.1:8000/sse"
    print("--- [Initializing Interactive MCP Agent Client] ---")
    print(f"🔄 Connecting to remote MCP Server at: {server_url}...")

    # AsyncExitStack securely manages the network connections
    async with AsyncExitStack() as stack:
        # 1. Establish SSE Transport and MCP Session
        try:
            sse_transport = await stack.enter_async_context(sse_client(url=server_url))
            read, write = sse_transport
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            print("✅ [Connection Established]: Ready to discover tools.\n")
        except Exception as e:
            print(f"❌ [Connection Error]: Failed to connect to server. Ensure it is running. Error: {e}")
            return

        # 2. Dynamic Tool Discovery
        mcp_registry = await session.list_tools()
        
        # Format tools into OpenAI's strict JSON schema
        tools_json = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "strict": True,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in mcp_registry.tools
        ]
        
        print(f"🛠️ [Tool Registry]: Discovered {len(tools_json)} remote tools.")
        print("-" * 65)
        print("💬 [Interactive Mode Started]. Type 'exit' or 'quit' to end the session.")
        print("-" * 65)

        # 3. Initialize Conversation State (Memory)
        chat_history = [
            {
                "role": "system",
                "content": "You are a professional and strict banking advisor. Use the available tools to assist the user accurately. Do not guess financial data."
            }
        ]

        # 4. The Interactive ReAct Loop (Stateful multi-turn conversation)
        while True:
            try:
                # Get user input from the terminal
                user_input = input("\n👤 You: ")
                if user_input.lower() in ['exit', 'quit']:
                    print("\n🔌 Ending session. Goodbye!")
                    break
                if not user_input.strip():
                    continue

                # Append user message to memory
                chat_history.append({"role": "user", "content": user_input})
                
                print("🤖 Agent is thinking...")
                
                # 4a. Trigger the LLM Brain
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=chat_history,
                    tools=tools_json if tools_json else None,
                )
                
                message_obj = response.choices[0].message
                # Save AI's response (or tool request) to memory
                chat_history.append(message_obj) 

                # 4b. Handle Tool Execution dynamically
                if message_obj.tool_calls:
                    for tool_call in message_obj.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)
                        
                        print(f"   ⚙️ [System Log]: Executing remote tool '{tool_name}' with args {tool_args}...")
                        
                        # Execute Remote Procedure Call (RPC) via MCP Session
                        tool_result = await session.call_tool(tool_name, arguments=tool_args)
                        observation = tool_result.content[0].text
                        
                        print(f"   📥 [Tool Observation]: {observation}")
                        
                        # Append the tool's output back to the conversation history
                        chat_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(observation),
                        })
                    
                    # 4c. Generate the final natural language response based on tool output
                    print("🤖 Agent is synthesizing the final response...")
                    final_response = await client.chat.completions.create(
                        model=MODEL,
                        messages=chat_history,
                    )
                    final_msg = final_response.choices[0].message.content
                    print(f"\n🤖 Agent: {final_msg}")
                    
                    # Append final answer to memory for the next turn
                    chat_history.append({"role": "assistant", "content": final_msg})
                    
                else:
                    # Normal conversational response (no tools needed)
                    print(f"\n🤖 Agent: {message_obj.content}")
                    
            # Graceful shutdown handling
            except KeyboardInterrupt:
                print("\n🔌 Session interrupted by user. Exiting...")
                break
            except Exception as e:
                print(f"\n❌ [Runtime Error]: {str(e)}")

if __name__ == "__main__":
    asyncio.run(run_interactive_agent())
