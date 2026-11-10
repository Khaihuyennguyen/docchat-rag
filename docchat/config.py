"""All settings in one place, read from environment variables."""
import os
from dataclasses import dataclass


@dataclass
class Settings:
    model: str = os.getenv("DOCCHAT_MODEL", "claude-sonnet-5-5")
    db_path: str = os.getenv("DOCCHAT_DB", ".docchat_db")
    chunk_size: int = int(os.getenv("DOCCHAT_CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("DOCCHAT_CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("DOCCHAT_TOP_K", "4"))
settings = Settings()