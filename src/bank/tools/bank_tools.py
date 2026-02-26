import os
import json
from langchain_core.tools import tool
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Resolve absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(BASE_DIR, "data")
faq_path = os.path.join(DATA_DIR, "US_Bank_FAQs.csv")
branch_path = os.path.join(DATA_DIR, "US_Bank_Branches.csv")

# 2. Initialize Embedding Model (Lightweight & Local)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. Load and Vectorize Data (In-memory FAISS RAG)
faq_retriever = None
branch_retriever = None

try:
    if os.path.exists(faq_path):
        faq_loader = CSVLoader(file_path=faq_path, encoding='utf-8')
        faq_vectorstore = FAISS.from_documents(faq_loader.load(), embeddings)

        faq_retriever = faq_vectorstore.as_retriever(search_kwargs={"k": 2})
        print("[INFO] RAG: FAQ Vector Store initialized successfully.")

    if os.path.exists(branch_path):
        branch_loader = CSVLoader(file_path=branch_path, encoding='utf-8')
        branch_vectorstore = FAISS.from_documents(branch_loader.load(), embeddings)

        branch_retriever = branch_vectorstore.as_retriever(search_kwargs={"k": 1})
        print("[INFO] RAG: Branch Vector Store initialized successfully.")
except Exception as e:
    print(f"[ERROR] Failed to initialize RAG Vector Stores: {e}")

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
    Finds the nearest physical bank branch using Semantic Vector Search (RAG).
    Handles natural language locations like "I'm near the windy city".
    """
    if not branch_retriever:
        return "System Error: Branch RAG engine is offline."

    # Perform similarity search
    docs = branch_retriever.invoke(user_location)
    
    if docs:
        return f"Nearest branch info retrieved from database:\n{docs[0].page_content}"
            
    return "Cannot find a branch near your location using semantic search."

@tool
def get_bank_faq(topic: str) -> str:
    """
    Retrieves official bank policies and FAQs using Semantic Vector Search (RAG).
    Understands intent even if exact keywords don't match.
    """
    if not faq_retriever:
        return "System Error: FAQ RAG engine is offline."

    # Perform similarity search for FAQs
    docs = faq_retriever.invoke(topic)
    
    if docs:
        context = "\n---\n".join([doc.page_content for doc in docs])
        return f"Policy context retrieved from database:\n{context}"
        
    return "I cannot find any policies related to your question in our vector database."