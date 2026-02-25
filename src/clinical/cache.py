import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Optional, Dict

class SemanticCache:
    """
    Enterprise-grade Semantic Cache with Multi-tenant isolation.
    Uses vector embeddings to detect similar queries and save LLM costs.
    """
    def __init__(self, similarity_threshold: float = 0.90):
        # Load a fast, lightweight embedding model for caching
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.threshold = similarity_threshold
        # In production, this would be Redis or Memcached. 
        # Using in-memory dict for demonstration.
        self.cache: Dict[str, list] = {}

    def _generate_namespace(self, patient_id: str) -> str:
        """Creates a secure hash namespace to prevent context leakage between patients."""
        return hashlib.sha256(patient_id.encode()).hexdigest()

    def check_cache(self, patient_id: str, query: str) -> Optional[str]:
        """
        Checks if a semantically similar query exists for this specific patient.
        """
        namespace = self._generate_namespace(patient_id)
        if namespace not in self.cache:
            return None

        query_vector = self.encoder.encode(query)
        
        # Scan patient's specific cache vault
        for cached_item in self.cache[namespace]:
            cached_vector = cached_item['vector']
            # Calculate Cosine Similarity
            similarity = np.dot(query_vector, cached_vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(cached_vector)
            )
            
            if similarity >= self.threshold:
                return cached_item['response']
                
        return None

    def store_cache(self, patient_id: str, query: str, response: str):
        """Stores a new query and response in the patient's isolated cache."""
        namespace = self._generate_namespace(patient_id)
        if namespace not in self.cache:
            self.cache[namespace] = []
            
        query_vector = self.encoder.encode(query)
        self.cache[namespace].append({
            'vector': query_vector,
            'response': response
        })

# Initialize a global singleton cache instance
clinical_cache = SemanticCache()
