"""
Model Context Protocol (MCP) Server Implementation.
This server exposes standard tools (e.g., weather lookup) that can be consumed 
dynamically by any MCP-compatible LLM client (like OpenAI Agents or Gemini).
"""
import json

class SimpleMCPServer:
    def __init__(self, name="Weather_MCP_Server"):
        self.name = name
        self.registered_tools = {}

    def register_tool(self, name, description, func):
        self.registered_tools[name] = {
            "description": description,
            "callable": func
        }

    def execute_tool(self, tool_name, **kwargs):
        if tool_name in self.registered_tools:
            return self.registered_tools[tool_name]["callable"](**kwargs)
        return json.dumps({"error": "Tool not found"})

# Define a mock function (as seen in the original Colab)
def get_current_temperature_by_city(city_name: str) -> str:
    """Mock weather API returning static data for demonstration."""
    return json.dumps({"city": city_name, "temperature": 20, "unit": "Celsius"})

def main():
    print("🚀 Starting MCP Server...")
    server = SimpleMCPServer()
    server.register_tool(
        "get_current_temperature_by_city", 
        "Get current temperature for a given city", 
        get_current_temperature_by_city
    )
    
    # Simulating an MCP Client request
    print(f"Server '{server.name}' is running. Available tools: {list(server.registered_tools.keys())}")
    result = server.execute_tool("get_current_temperature_by_city", city_name="Hanoi")
    print(f"Tool Execution Result: {result}")

if __name__ == "__main__":
    main()
