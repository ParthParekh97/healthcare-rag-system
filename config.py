import os
from dotenv import load_dotenv

load_dotenv()


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

TOP_K_RETRIEVAL = 5
SIMILARITY_THRESHOLD = 0.35

LLM_MODEL = "gemini-2.5-flash"
LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
LLM_TEMPERATURE = 0.1
MAX_RESPONSE_TOKENS = 1024

GROUNDING_STRICTNESS = 0.55

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "sample_docs")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "data", "index")
