import json
import numpy as np
import redis
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.query import Query
from redis.commands.search.index_definition import IndexDefinition, IndexType
from sentence_transformers import SentenceTransformer

# Initialize the embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

try:
    # Requires Redis Stack (RediSearch module enabled)
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)
    redis_client.ping()
    print("[INFO] Redis connection established for High-Performance Semantic Cache.")
except Exception as e:
    print(f"[WARNING] Redis connection failed: {e}. Proceeding without caching.")
    redis_client = None

class ClinicalSemanticCache:
    def __init__(self, threshold: float = 0.65, index_name: str = "idx:clinical_cache"):
        """
        Initializes the Semantic Cache utilizing RediSearch for O(1) vector lookups.
        """
        self.threshold = threshold
        self.index_name = index_name
        self._initialize_redis_index()

    def _initialize_redis_index(self):
        """Creates the Vector Search Index in Redis if it does not already exist."""
        if not redis_client:
            return
            
        try:
            redis_client.ft(self.index_name).info()
            print("[INFO] Redis Vector Index already exists.")
        except redis.exceptions.ResponseError:
            print("[INFO] Creating new Redis Vector Index...")
            # Define schema: isolating by patient_id, searching by embedding
            schema = (
                TextField("patient_id"),
                TextField("response"),
                VectorField("embedding", "FLAT", {
                    "TYPE": "FLOAT32",
                    "DIM": 384,  # MiniLM dimension
                    "DISTANCE_METRIC": "COSINE"
                })
            )
            definition = IndexDefinition(prefix=["clinical_cache:"], index_type=IndexType.HASH)
            redis_client.ft(self.index_name).create_index(fields=schema, definition=definition)

    def check_cache(self, patient_id: str, query: str) -> str:
        """
        Executes a K-Nearest Neighbors (KNN) vector search natively inside Redis.
        Ensures strict tenant isolation via the patient_id filter.
        """
        if not redis_client:
            return None

        query_vector = embedding_model.encode(query).astype(np.float32).tobytes()
        
        # Construct RediSearch Query: Filter by patient_id, then calculate KNN
        q = (
            Query(f"(@patient_id:{{{patient_id}}})=>[KNN 1 @embedding $vec AS score]")
            .sort_by("score")
            .return_fields("response", "score")
            .dialect(2)
        )
        
        try:
            results = redis_client.ft(self.index_name).search(q, {"vec": query_vector})
            
            if results.docs:
                best_match = results.docs[0]
                # Note: RediSearch COSINE distance is (1 - cosine_similarity)
                similarity = 1 - float(best_match.score)
                
                print(f"[DEBUG] Cache Lookup - Similarity Score: {similarity:.4f}")
                
                if similarity >= self.threshold:
                    print("[INFO] Semantic Cache Hit! Retrieving instantly.")
                    return best_match.response
        except Exception as e:
            print(f"[ERROR] Cache retrieval failed: {e}")
            
        return None

    def store_cache(self, patient_id: str, query: str, response: str) -> None:
        """
        Embeds the query and stores the transaction in the vector database.
        Includes a 24-hour TTL for strict clinical compliance.
        """
        if not redis_client:
            return

        # Sanitize query to form a valid Redis key
        safe_query = query.replace(" ", "_").replace(":", "")[:50]
        redis_key = f"clinical_cache:{patient_id}:{safe_query}"
        query_vector = embedding_model.encode(query).astype(np.float32).tobytes()
        
        # Map data for RediSearch ingestion
        mapping = {
            "patient_id": patient_id,
            "response": response,
            "embedding": query_vector
        }
        
        # Store Hash and set Time-To-Live (86400 seconds = 24h)
        redis_client.hset(redis_key, mapping=mapping)
        redis_client.expire(redis_key, 86400)
        print(f"[INFO] Interaction cached successfully for tenant: {patient_id}")

# Export singleton instance
clinical_cache = ClinicalSemanticCache()
