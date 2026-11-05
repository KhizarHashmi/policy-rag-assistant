"""Provider-agnostic chat model factory. Swap providers with LLM_PROVIDER, no code changes."""
import warnings
from functools import lru_cache

from langchain_core.language_models import BaseChatModel

from .config import settings

# Some Gemini models pin sampling params; the provider ignores temperature and warns on every call.
warnings.filterwarnings("ignore", message=".*fixed sampling defaults.*")


@lru_cache(maxsize=4)
def get_llm(temperature: float = 0.0) -> BaseChatModel:
    if settings.provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            temperature=temperature,
            thinking_level=settings.gemini_thinking_level,
            max_retries=6,
            timeout=120,  # seconds; a hung request must not stall the pipeline
        )
    if settings.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.anthropic_model, temperature=temperature, max_retries=4, timeout=120
        )
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.provider!r}")
