'''Central configuration via environment variables / settings.'''

from pydantic import BaseModel
import os

class Settings(BaseModel):
    # Paths
    persist_dir: str = os.getenv("PERSIST_DIR", "./data/chroma_store")
    runs_dir: str = os.getenv("RUNS_DIR", "./data/runs")

    # Chroma collections
    hist_collection: str = os.getenv("HIST_COLLECTION", "historical_tenders_v1")
    gen_collection: str = os.getenv("GEN_COLLECTION", "generated_repo")

    # Retrieval & routing thresholds
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "3"))
    strong_threshold: float = float(os.getenv("STRONG_THRESHOLD", "0.70"))
    weak_threshold: float = float(os.getenv("WEAK_THRESHOLD", "0.60"))

    # Caching toggles (dev)
    enable_cache: bool = os.getenv("ENABLE_CACHE", "true").lower() == "true"

    # Embeddings config
    embeddings_provider: str = os.getenv("EMBEDDINGS_PROVIDER", "openai")  # openai 
    embeddings_model: str = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-large")

    # Domain classifier + Generator + Softner + Verify config
    llm_provider: str = os.getenv("PROVIDER", "openai")  # openai for now
    llm_model: str = os.getenv("MODEL", "gpt-4o-mini")


settings = Settings()
