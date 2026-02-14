from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    vector_store_dir: str


def get_settings() -> Settings:
    return Settings(
        vector_store_dir=os.getenv("SCISEARCH_VECTOR_STORE_DIR", "data/vector_store"),
    )

