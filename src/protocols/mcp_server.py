from fastmcp import FastMCP
from src.bank.tools.bank_tools import calculate_dti, search_nearest_branch, get_bank_faq

# Initialize the FastMCP Server
mcp = FastMCP("Enterprise-Bank-Server")

# ==========================================
# 1. RESOURCES: Static Context for the LLM
# ==========================================
@mcp.resource("bank://compliance/core_principles")
def get_compliance_guidelines() -> str:
    """
    Provides static, read-only compliance guidelines.
    The LLM fetches this URI to understand business boundaries before acting.
    """
    return (
        "BANK COMPLIANCE POLICY 2026:\n"
        "1. Never guarantee loan approval without a DTI check.\n"
        "2. Ensure all financial data is handled securely.\n"
        "3. Protect user data privacy at all costs."
    )

# ==========================================
# 2. PROMPTS: Standardized Execution Templates
# ==========================================
@mcp.prompt()
def formal_loan_assessment(customer_name: str) -> str:
    """
    A standardized prompt template for generating official loan reports.
    Instructs the Agent on exactly how to format its final response.
    """
    return (
        f"Perform a formal loan risk assessment for {customer_name}. "
        "Use the 'compute_dti' tool to calculate the risk based on the user's input. "
        "Format your final response as an official bank letter, including the DTI ratio, "
        "the risk tier, and a standard compliance disclaimer."
    )

# ==========================================
# 3. TOOLS: Executable Functions
# ==========================================
@mcp.tool()
def compute_dti(monthly_income: float, monthly_debt: float) -> str:
    """Calculates the Debt-to-Income (DTI) ratio for financial risk assessment."""
    return calculate_dti.invoke({"monthly_income": monthly_income, "monthly_debt": monthly_debt})

@mcp.tool()
def fetch_bank_faq(topic: str) -> str:
    """Retrieves official bank policies using semantic RAG search."""
    return get_bank_faq.invoke({"topic": topic})

if __name__ == "__main__":
    # Launch the server using Server-Sent Events (SSE) transport on Port 8000
    print("--- [Initializing FastMCP SSE Server on Port 8000] ---")
    mcp.run(transport="sse", port=8000)
