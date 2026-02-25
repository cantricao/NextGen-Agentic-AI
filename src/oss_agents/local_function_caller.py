"""
Offline Function Calling with Open-Source LLMs.
Demonstrates how to parse and execute function calls locally without proprietary APIs.
"""
import json

def get_random_number() -> int:
    """Returns a random number (mocked for deterministic testing)."""
    return 6

def simulate_oss_llm_response(prompt: str) -> str:
    """
    Simulates the text-generation of an Open-Source LLM (like Hermes or Llama-3)
    that has been fine-tuned to output JSON tool calls.
    """
    # Mocking the LLM's raw output string
    return '{"name": "get_random_number", "arguments": {}}'

def main():
    print("🛠️ Testing Open-Source Function Calling...")
    user_prompt = "Give a random number"
    print(f"User: {user_prompt}")
    
    # 1. Get raw string from OSS LLM
    llm_output = simulate_oss_llm_response(user_prompt)
    print(f"LLM Raw Output: {llm_output}")
    
    # 2. Parse and execute locally
    try:
        tool_call = json.loads(llm_output)
        if tool_call.get("name") == "get_random_number":
            result = get_random_number()
            print(f"⚙️ Function Executed. Result: {result}")
            print(f"🤖 Final Answer: The random number generated is {result}.")
    except Exception as e:
        print(f"Failed to parse tool call: {e}")

if __name__ == "__main__":
    main()
