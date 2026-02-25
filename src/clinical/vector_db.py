import os
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

# Initialize lightweight local embedding model
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Resolve path for local ChromaDB persistence
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHROMA_PATH = os.path.join(BASE_DIR, "data", "clinical_chroma_db")

def get_clinical_vector_db():
    """Initializes and returns the Chroma Vector Store."""
    vector_store = Chroma(
        collection_name="patient_records",
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )
    return vector_store

def seed_database():
    """Seeds the database with mock medical records if it is empty."""
    db = get_clinical_vector_db()
    
    # Check if database already contains documents
    existing_docs = db.get()
    if len(existing_docs['ids']) > 0:
        return db

    print("[INFO] Initializing Medical Vector Database...")
    
    # Mock documents WITH METADATA for Multi-tenant isolation
    medical_records = [
        Document(
            page_content="Patient presents with severe influenza A symptoms. Prescribed Oseltamivir (Tamiflu) 75mg twice daily for 5 days. Patient has a known allergy to Penicillin.",
            metadata={"patient_id": "PT-88902", "doc_type": "consultation_note", "date": "2026-02-15"}
        ),
        Document(
            page_content="Blood test results: High LDL cholesterol (160 mg/dL). Recommended dietary changes and started on Atorvastatin 20mg daily.",
            metadata={"patient_id": "PT-88902", "doc_type": "lab_result", "date": "2026-02-20"}
        ),
        Document(
            page_content="Patient reports chronic lower back pain. MRI shows mild disc herniation at L4-L5. Prescribed physical therapy and Ibuprofen 400mg PRN.",
            metadata={"patient_id": "PT-55101", "doc_type": "mri_report", "date": "2026-01-10"}
        )
    ]
    
    db.add_documents(medical_records)
    print("[INFO] Medical records successfully seeded into ChromaDB.")
    return db

# Execute seed function upon module import
clinical_db = seed_database()
