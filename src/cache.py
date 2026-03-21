import redis
import json
import os
import uuid
import hashlib
from typing import Optional, Dict, Any, List
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
LOCAL_CACHE_DIR = os.path.join(os.getcwd(), ".stroke_cache")
LOCAL_CACHE_FILE = os.path.join(LOCAL_CACHE_DIR, "cache.json")
# We use a dedicated vector store just for mapping questions semantically
VECTOR_CACHE_DIR = os.path.join(LOCAL_CACHE_DIR, "vector_db")

class StrokeCache:
    """Hybrid Semantic Cache using Vector Search (Chroma) and Key-Value Storage (Redis/File)."""
    
    def __init__(self, url: str = REDIS_URL):
        self.redis_available = False
        self.local_available = False
        self.threshold = 0.2  # Semantic distance threshold (lower is stricter)
        
        # 1. Initialize Semantic Vector Engine
        try:
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            self.vector_store = Chroma(
                collection_name="semantic_cache_index",
                persist_directory=VECTOR_CACHE_DIR,
                embedding_function=self.embeddings
            )
            print(f"--- INFO: Semantic Cache Index ready at {VECTOR_CACHE_DIR} ---")
        except Exception as e:
            print(f"--- WARNING: Semantic Indexing disabled: {str(e)} ---")
            self.vector_store = None

        # 2. Initialize Redis
        try:
            self.redis = redis.from_url(url, socket_timeout=1)
            self.redis.ping()
            self.redis_available = True
            print(f"--- INFO: Connected to Redis Cache at {url} ---")
        except redis.ConnectionError:
            print("--- WARNING: Redis not reachable. Using local storage for cache content. ---")
            
        # 3. Initialize Local File Fallback
        try:
            if not os.path.exists(LOCAL_CACHE_DIR):
                os.makedirs(LOCAL_CACHE_DIR)
            if not os.path.exists(LOCAL_CACHE_FILE):
                with open(LOCAL_CACHE_FILE, "w") as f:
                    json.dump({}, f)
            self.local_available = True
        except OSError:
            print("--- ERROR: Local cache storage failed to initialize ---")

    def _generate_vector_id(self, question: str) -> str:
        """Create a stable ID for the question vector to avoid duplicates."""
        return hashlib.md5(question.strip().lower().encode()).hexdigest()

    def get_response(self, question: str) -> Optional[Dict[str, Any]]:
        """Semantic retrieval: Finds the best matching past question."""
        if not self.vector_store:
            return None

        # Search for semantically similar questions
        # score is L2 distance (lower is closer)
        try:
            results = self.vector_store.similarity_search_with_score(question, k=1)
            
            if results:
                doc, score = results[0]
                if score < self.threshold:
                    cache_key = doc.metadata.get("cache_key")
                    print(f"--- SEMANTIC CACHE HIT: Match found with score {score:.4f} ---")
                    return self._fetch_from_storage(cache_key)
                else:
                    print(f"--- CACHE MISS: Nearest match score {score:.4f} (Threshold: {self.threshold}) ---")
        except Exception as e:
            print(f"--- ERROR: Semantic search failed: {str(e)} ---")
        
        return None

    def set_response(self, question: str, response: Dict[str, Any], expire: int = 86400):
        """Save to semantic index and storage."""
        # 1. Generate unique key for content
        cache_key = str(uuid.uuid4())
        
        # 2. Update Semantic Index
        if self.vector_store:
            try:
                self.vector_store.add_texts(
                    texts=[question],
                    metadatas=[{"cache_key": cache_key}],
                    ids=[self._generate_vector_id(question)]
                )
            except Exception as e:
                print(f"--- ERROR: Could not update semantic index: {str(e)} ---")

        # 3. Save Payload
        payload = {
            "generation": response.get("generation"),
            "documents": [
                {
                    "page_content": d.page_content if hasattr(d, 'page_content') else d.get('page_content', ''),
                    "metadata": d.metadata if hasattr(d, 'metadata') else d.get('metadata', {})
                } for d in response.get("documents", [])
            ]
        }
        self._save_to_storage(cache_key, payload, expire)

    def _fetch_from_storage(self, key: str) -> Optional[Dict[str, Any]]:
        # Redis First
        if self.redis_available:
            try:
                data = self.redis.get(key)
                if data:
                    return self._reconstruct_documents(json.loads(data))
            except redis.ConnectionError:
                self.redis_available = False
        
        # File Fallback
        if self.local_available:
            try:
                with open(LOCAL_CACHE_FILE, "r") as f:
                    data = json.load(f)
                    if key in data:
                        return self._reconstruct_documents(data[key])
            except:
                pass
        return None

    def _save_to_storage(self, key: str, payload: Dict[str, Any], expire: int):
        # Save to Redis
        if self.redis_available:
            try:
                self.redis.set(key, json.dumps(payload), ex=expire)
                print(f"--- CACHE SAVE (REDIS): Content stored under {key[:8]} ---")
            except redis.ConnectionError:
                self.redis_available = False
        
        # Save to Local File
        if self.local_available:
            try:
                with open(LOCAL_CACHE_FILE, "r") as f:
                    data = json.load(f)
                data[key] = payload
                with open(LOCAL_CACHE_FILE, "w") as f:
                    json.dump(data, f, indent=4)
                print(f"--- CACHE SAVE (LOCAL): Content stored under {key[:8]} ---")
            except:
                pass

    def _reconstruct_documents(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert cached dictionaries back into LangChain Document objects."""
        if "documents" in data and isinstance(data["documents"], list):
            data["documents"] = [
                Document(page_content=d.get("page_content", ""), metadata=d.get("metadata", {}))
                for d in data["documents"]
            ]
        return data

cache = StrokeCache()
