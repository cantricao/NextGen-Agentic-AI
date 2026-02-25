import os
import json
import pandas as pd
from langchain_core.tools import tool

# Resolve the absolute path to the data directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Load CSV files into memory during system startup
faq_path = os.path.join(DATA_DIR, "US_Bank_FAQs.csv")
branch_path = os.path.join(DATA_DIR, "US_Bank_Branches.csv")

faq_df = pd.read_csv(faq_path) if os.path.exists(faq_path) else pd.DataFrame()
branch_df = pd.read_csv(branch_path) if os.path.exists(branch_path) else pd.DataFrame()

@tool
def calculate_dti(monthly_income: float, monthly_debt: float) -> str:
    """Calculates the Debt-to-Income (DTI) ratio for a customer."""
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
    Reads real data from US_Bank_Branches.csv.
    """
    if branch_df.empty:
        return "Branch data system is currently under maintenance or missing CSV file."

    location_key = user_location.lower().strip()
    
    # Scan for the location keyword across all columns in the CSV
    mask = branch_df.apply(lambda row: row.astype(str).str.lower().str.contains(location_key).any(), axis=1)
    results = branch_df[mask]

    if not results.empty:
        top_branch = results.iloc[0].to_dict()
        return f"Found nearest branch: {top_branch}."
            
    return "Cannot find a branch near your location. Please contact our hotline at 1900-xxxx."

@tool
def get_bank_faq(topic: str) -> str:
    """
    Retrieves official bank policies, interest rates, and general FAQs.
    Reads real data from US_Bank_FAQs.csv.
    """
    if faq_df.empty:
        return "Policy data system is currently under maintenance or missing CSV file."

    topic_key = topic.lower().strip()
    
    # Scan for the topic keyword across all columns in the CSV
    mask = faq_df.apply(lambda row: row.astype(str).str.lower().str.contains(topic_key).any(), axis=1)
    results = faq_df[mask]

    if not results.empty:
        answer = results.iloc[0].to_dict()
        return f"Policy information: {answer}"
        
    return "I cannot find
