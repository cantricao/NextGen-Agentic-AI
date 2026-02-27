import asyncio
import json
import os
from contextlib import AsyncExitStack
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.sse import sse_client
from openai import OpenAI

# Load environment variables (ensure GEMINI_API_KEY or GOOGLE_API_KEY is set in .env)
load_dotenv()

# Initialize the OpenAI client pointing to Gemini's compatibility endpoint
client = OpenAI(
    api_key=os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY")), 
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
model = "gemini-2.5-flash"


class ConnectionManager:
    """
    Manages multiple SSE connections to different MCP Servers simultaneously.
    Utilizes AsyncExitStack for safe and clean resource teardown.
    """
    def __init__(self, sse_server_map):
        self.sse_server_map = sse_server_map
        self.sessions = {}
        self.exit_stack = AsyncExitStack()

    async def initialize(self):
        """Initializes SSE connections for all registered servers."""
        for server_name, url in self.sse_server_map.items():
            # Establish SSE transport layer
            sse_transport = await self.exit_stack.enter_async_context(
                sse_client(url=url)
            )
            read, write = sse_transport
            
            # Establish MCP Client Session over the transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.sessions[server_name] = session
            print(f"✅ [ConnectionManager]: Connected to {server_name} at {url}")

    async def list_tools(self):
        """Consolidates tools from all connected MCP servers."""
        tool_map = {}
        consolidated_tools = []
        for server_name, session in self.sessions.items():
            tools = await session.list_tools()
            # Map each tool to its origin server for correct routing later
            tool_map.update({tool.name: server_name for tool in tools.tools})
            consolidated_tools.extend(tools.tools)
        return tool_map, consolidated_tools

    async def call_tool(self, tool_name, arguments, tool_map):
        """Routes the tool execution request to the correct specific server."""
        server_name = tool_map.get(tool_name)
        if not server_name:
            print(f"❌ [Error]: Tool '{tool_name}' not found in any connected server.")
            return

        session = self.sessions.get(server_name)
        if session:
            result = await session.call_tool(tool_name, arguments=arguments)
            return result.content[0].text

    async def close(self):
        """Gracefully closes all active network connections."""
        await self.exit_stack.aclose()
        print("🔌 [ConnectionManager]: All connections closed safely.")


async def chat(input_messages, tool_map, tools, max_turns=3, connection_manager=None):
    """
    Core ReAct execution loop handling raw OpenAI message structures.
    Yields intermediate steps (Tool Calls/Observations) for real-time streaming.
    """
    chat_messages = input_messages[:]
    
    for _ in range(max_turns):
        # 1. Trigger the LLM Brain
        result = client.chat.completions.create(
            model=model,
            messages=chat_messages,
            tools=tools,
        )
        
        message_obj = result.choices[0].message

        # 2. Check if the LLM decided to use a tool
        if result.choices[0].finish_reason == "tool_calls" or hasattr(message_obj, 'tool_calls') and message_obj.tool_calls:
            chat_messages.append(message_obj)
            
            # Loop through all requested tools (supports parallel tool calling)
            for tool_call in message_obj.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                server_name = tool_map.get(tool_name, "Unknown")

                # Yield action log
                log_message = f"🛠️ **Tool Call Executed** \n* **Tool:** `{tool_name}`\n* **Server:** `{server_name}`\n* **Payload:** `{json.dumps(tool_args)}`"
                yield {"role": "assistant", "content": log_message}

                # Execute remote procedure call
                observation = await connection_manager.call_tool(tool_name, tool_args, tool_map)
                
                # Yield observation log
                log_message = f"📥 **Tool Observation** \n* **Result:** `{observation}`\n---"
                yield {"role": "assistant", "content": log_message}

                # Append the raw output back into the conversation history
                chat_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(observation),
                })
        else:
            # 3. No tools requested, yield final natural language response
            yield {"role": "assistant", "content": message_obj.content}
            return

    # Fallback if max turns exceeded
    yield {"role": "assistant", "content": "⚠️ [System]: Max reasoning turns reached."}


if __name__ == "__main__":
    # Map to our local Enterprise Bank MCP Server
    sse_server_map = {
        "enterprise_bank_mcp": "http://127.0.0.1:8000/sse",
    }

    async def main():
        print("--- [Starting Native OpenAI MCP Client Integration] ---")
        connection_manager = ConnectionManager(sse_server_map)
        await connection_manager.initialize()

        # Retrieve and format tools into OpenAI's strict JSON schema
        tool_map, tool_objects = await connection_manager.list_tools()
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
            for tool in tool_objects
        ]

        # Define the banking intent
        input_messages = [
            {
                "role": "system",
                "content": "You are a senior financial advisor. Use available tools to analyze client data.",
            },
            {"role": "user", "content": "My monthly income is $12000 and my debts are $3500. Can you calculate my DTI?"},
        ]

        print(f"\n👤 User: {input_messages[1]['content']}\n")

        # Stream the thought process and final answer
        async for response in chat(
            input_messages,
            tool_map,
            tools=tools_json,
            connection_manager=connection_manager,
        ):
            print(response["content"])

        # Teardown
        print("\n--- [Execution Complete] ---")
        await connection_manager.close()

    # Run the event loop
    asyncio.run(main())
