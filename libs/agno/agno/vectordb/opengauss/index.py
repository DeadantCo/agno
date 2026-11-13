from typing import Optional

from pydantic import BaseModel


class Ivfflat(BaseModel):
    name: Optional[str] = None
    lists: int = 100
    probes: int = 10
    dynamic_lists: bool = True


class HNSW(BaseModel):
    name: Optional[str] = None
    m: int = 16
    ef_search: int = 64
    ef_construction: int = 200
