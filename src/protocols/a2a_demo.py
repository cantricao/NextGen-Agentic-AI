"""
Google Agent-to-Agent (A2A) Communication Protocol.
Demonstrates dynamic agent discovery and task delegation using a Coordinator pattern.
"""

class MockLlmAgent:
    """A mock representation of the Google A2A LlmAgent class."""
    def __init__(self, name, model, description, sub_agents=None):
        self.name = name
        self.model = model
        self.description = description
        self.sub_agents = sub_agents or []

    def handle_task(self, task: str):
        print(f"[{self.name}] Analyzing task: '{task}'")
        if "greet" in task.lower() and self.sub_agents:
            print(f"[{self.name}] Delegating to sub-agent...")
            self.sub_agents[0].handle_task(task)
        else:
            print(f"[{self.name}] Executing task independently.")

def main():
    print("🌐 Initializing Google A2A Ecosystem...")
    
    # 1. Define Sub-Agents
    greeter = MockLlmAgent(
        name="GreeterAgent", 
        model="gemini-2.5-flash", 
        description="I handle user greetings."
    )
    task_doer = MockLlmAgent(
        name="TaskDoerAgent", 
        model="gemini-2.5-flash", 
        description="I execute complex backend tasks."
    )
    
    # 2. Define Parent Coordinator
    coordinator = MockLlmAgent(
        name="Coordinator",
        model="gemini-2.5-pro",
        description="I coordinate greetings and tasks.",
        sub_agents=[greeter, task_doer]
    )
    
    # 3. Simulate A2A Communication
    print("\\n--- Simulating User Request ---")
    coordinator.handle_task("Please greet the new customer.")

if __name__ == "__main__":
    main()
