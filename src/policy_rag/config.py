"""Runtime configuration, read once from environment variables (or a .env file)."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = Path(os.getenv("DOCS_DIR", PROJECT_ROOT / "data" / "policies"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", PROJECT_ROOT / ".chroma"))
EVAL_DIR = PROJECT_ROOT / "eval"


@dataclass(frozen=True)
class Settings:
    provider: str = os.getenv("LLM_PROVIDER", "gemini")  # "gemini" | "anthropic" (generation)
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    gemini_thinking_level: str = os.getenv("GEMINI_THINKING_LEVEL", "low")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "768"))
    chunk_max_chars: int = int(os.getenv("CHUNK_MAX_CHARS", "900"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    collection: str = os.getenv("COLLECTION", "policies")


settings = Settings()
