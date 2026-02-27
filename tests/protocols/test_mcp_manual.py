import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client

async def run_manual_infrastructure_test():
    """
    Integration Test Suite for the FastMCP SSE Server.
    Strictly tests the network and protocol layers WITHOUT using any LLM.
    Ensures that the server correctly exposes Resources, Prompts, and Tools.
    """
    server_url = "http://127.0.0.1:8000/sse"
    print(f"--- [Initializing Manual MCP Infrastructure Test] ---")
    print(f"🔄 Attempting to connect to {server_url}...\n")

    try:
        # Establish the SSE connection to the remote server
        async with sse_client(server_url) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the MCP handshake
                await session.initialize()
                print("✅ [Handshake]: Connection securely established.\n")

                # ---------------------------------------------------------
                # TEST 1: RESOURCE RETRIEVAL (Read-only data)
                # ---------------------------------------------------------
                print("🧪 [Test 1]: Fetching Bank Compliance Resource...")
                resource = await session.read_resource("bank://compliance/core_principles")
                
                # Verify that data was actually returned
                assert len(resource.contents) > 0, "Resource contents should not be empty."
                print(f"   -> [PASS] Successfully fetched {len(resource.contents[0].text)} characters.")

                # ---------------------------------------------------------
                # TEST 2: TOOL DISCOVERY (Checking exposed endpoints)
                # ---------------------------------------------------------
                print("\n🧪 [Test 2]: Fetching Available Tools Registry...")
                mcp_registry = await session.list_tools()
                
                found_tools = [t.name for t in mcp_registry.tools]
                print(f"   -> [PASS] Found {len(found_tools)} tools: {', '.join(found_tools)}")
                
                # Verify that our specific tools are registered
                assert "compute_dti" in found_tools, "Tool 'compute_dti' is missing from server."
                assert "fetch_bank_faq" in found_tools, "Tool 'fetch_bank_faq' is missing from server."

                # ---------------------------------------------------------
                # TEST 3: TOOL EXECUTION (RPC with Hardcoded Arguments)
                # ---------------------------------------------------------
                print("\n🧪 [Test 3]: Executing 'compute_dti' via Remote Procedure Call...")
                
                # Hardcode the arguments to bypass the LLM completely
                mock_payload = {
                    "monthly_income": 10000, 
                    "monthly_debt": 2500
                } 
                
                # Trigger the remote execution
                tool_result = await session.call_tool("compute_dti", arguments=mock_payload)
                
                print(f"   -> [PASS] Server returned result: {tool_result.content[0].text}\n")
                
                # ---------------------------------------------------------
                print("🎉 [SYSTEM CHECK]: ALL INFRASTRUCTURE TESTS PASSED.")

    except AssertionError as ae:
        print(f"❌ [ASSERTION FAILED]: {str(ae)}")
    except Exception as e:
        print(f"❌ [NETWORK/PROTOCOL ERROR]: {str(e)}")
        print("💡 Hint: Did you forget to run 'python src/protocols/mcp_server.py' in a separate terminal?")

if __name__ == "__main__":
    # Execute the asynchronous test suite
    asyncio.run(run_manual_infrastructure_test())
