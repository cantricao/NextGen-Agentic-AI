from langchain_core.tools import tool
import json

@tool
def calculate_dti(monthly_income: float, monthly_debt: float) -> str:
    """
    Calculates the Debt-to-Income (DTI) ratio for a customer.
    The Loan Agent MUST use this tool to evaluate creditworthiness before approving a loan.
    
    Args:
        monthly_income (float): Total monthly income in local currency.
        monthly_debt (float): Total monthly debt payments.
        
    Returns:
        str: A JSON string containing the DTI ratio and a risk assessment.
    """
    if monthly_income <= 0:
        return json.dumps({"error": "Income must be greater than zero."})
    
    dti = round(monthly_debt / monthly_income, 2)
    risk_level = "Low" if dti < 0.36 else "Moderate" if dti <= 0.43 else "High"
    
    return json.dumps({
        "dti_ratio": dti,
        "risk_level": risk_level,
        "advice": "Recommend approval" if risk_level != "High" else "Require further manual review"
    })

@tool
def search_nearest_branch(user_location: str) -> str:
    """
    Finds the nearest physical bank branch based on the user's current city or location.
    
    Args:
        user_location (str): The current city, province, or address of the user.
        
    Returns:
        str: Address and distance to the nearest branch.
    """
    # Mock database mapping locations to branches
    database = {
        "hanoi": "Hoan Kiem Branch, 123 Ly Thai To, Hanoi (Distance: 2.1 km)",
        "ho chi minh": "District 1 Branch, 456 Nguyen Hue, HCMC (Distance: 1.5 km)",
        "da nang": "Hai Chau Branch, 789 Bach Dang, Da Nang (Distance: 3.0 km)"
    }
    
    location_key = user_location.lower()
    for key, branch in database.items():
        if key in location_key:
            return f"The nearest branch is: {branch}."
            
    return "Cannot find a branch near your location. Please contact our hotline at 1900-xxxx."

@tool
def get_bank_faq(topic: str) -> str:
    """
    Retrieves official bank policies, interest rates, and general FAQs.
    
    Args:
        topic (str): The specific topic to look up (e.g., 'interest rate', 'credit card fee').
        
    Returns:
        str: Information regarding the requested policy.
    """
    # In production, this would query FAISS or ChromaDB. 
    # Here we use a mock retrieval for demonstration.
    topic = topic.lower()
    if "interest" in topic or "rate" in topic:
        return "Current loan interest rate is 6.5% p.a. for the first 12 months, and floating thereafter."
    elif "fee" in topic or "card" in topic:
        return "Annual credit card fee is $50, waived if annual spending exceeds $5,000."
    return "Please visit our website for more detailed policies."
