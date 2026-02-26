import json
import numpy as np
import redis
from sentence_transformers import SentenceTransformer

# Initialize the lightweight local embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Redis connection for local Colab environment
try:
    # Using decode_responses=True automatically decodes bytes to strings
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()
    print("[INFO] Redis connection established for Semantic Cache.")
except Exception as e:
    print(f"[WARNING] Redis connection failed: {e}. Proceeding without caching.")
    redis_client = None

class ClinicalSemanticCache:
    # Lowered threshold to 0.65 for better semantic matching with MiniLM
    def __init__(self, threshold: float = 0.65):
        """
        Initializes the Semantic Cache with a cosine similarity threshold.
        Queries scoring above this threshold will trigger a cache hit.
        """
        self.threshold = threshold

    def _get_namespace_key(self, patient_id: str) -> str:
        """Generates a secure, isolated Redis Hash key per patient."""
        return f"clinical_cache:{patient_id}"

    def cosine_similarity(self, v1: list, v2: list) -> float:
        """Calculates the cosine similarity between two vectors."""
        vec1, vec2 = np.array(v1), np.array(v2)
        norm_1 = np.linalg.norm(vec1)
        norm_2 = np.linalg.norm(vec2)
        
        if norm_1 == 0 or norm_2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm_1 * norm_2))

    def check_cache(self, patient_id: str, query: str) -> str | None:
        """
        Scans the Redis Hash for the patient to find semantically similar queries.
        Returns the cached response if similarity exceeds the threshold.
        """
        if not redis_client:
            print("[ERROR] Cannot check cache: Redis client is offline.")
            return None

        redis_key = self._get_namespace_key(patient_id)
        query_vector = embedding_model.encode(query).tolist()
        
        # Retrieve all cached query-response pairs for this specific patient
        cached_entries = redis_client.hgetall(redis_key)
        
        best_score = 0.0
        best_response = None

        # Compare the incoming query vector against all cached vectors in the namespace
        for cached_query_text, json_payload in cached_entries.items():
            try:
                data = json.loads(json_payload)
                cached_vector = data.get("embedding", [])
                
                score = self.cosine_similarity(query_vector, cached_vector)
                if score > best_score:
                    best_score = score
                    best_response = data.get("response")
            except json.JSONDecodeError:
                continue
                
        print(f"[DEBUG] Highest similarity score found: {best_score:.4f}")
        
        if best_score >= self.threshold:
            print(f"[DEBUG] Semantic Cache Hit! Similarity: {best_score:.4f}")
            return best_response
            
        return None

    def store_cache(self, patient_id: str, query: str, response: str) -> None:
        """
        Embeds the user query and stores it along with the LLM response 
        in the Redis Hash, scoped to the patient ID.
        """
        if not redis_client:
            print("[ERROR] Cannot store cache: Redis client is offline. Please start Redis server.")
            return

        redis_key = self._get_namespace_key(patient_id)
        query_vector = embedding_model.encode(query).tolist()
        
        # Serialize the vector and response into a JSON payload
        payload = json.dumps({
            "embedding": query_vector,
            "response": response
        })
        
        # Store securely in the patient's specific Redis Hash
        redis_client.hset(redis_key, query, payload)
        
        # Set a 24-hour Time-To-Live (TTL) for compliance with EHR data policies
        redis_client.expire(redis_key, 86400)
        print(f"[INFO] Successfully cached new interaction for patient: {patient_id}")

# Export a globally accessible singleton instance
clinical_cache = ClinicalSemanticCache()