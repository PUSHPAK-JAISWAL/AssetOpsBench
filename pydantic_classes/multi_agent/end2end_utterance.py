from typing import List, Union
from pydantic import BaseModel

class KnowledgeQueryItem(BaseModel):
    id: int
    type: str
    text: str
    category: str
    characteristic_form: str

class DeterministicItem(BaseModel):
    id: int
    type: str
    characteristic_form: str
    deterministic: bool
    text: str

class ConditionalItem(BaseModel):
    id: int
    type: str
    characteristic_form: str
    deterministic: bool
    text: str

class AnomalyCheckItem(BaseModel):
    id: int
    type: str
    text: str
    characteristic_form: str

class Dataset(BaseModel):
    items: List[Union[
        KnowledgeQueryItem,
        DeterministicItem,
        ConditionalItem,
        AnomalyCheckItem
    ]]


# KnowledgeQueryItem: For items with category field (IDs 501-520)

# DeterministicItem: For items with deterministic=True (IDs 601-602)

# ConditionalItem: For items with deterministic=False (IDs 603-620)

# AnomalyCheckItem: For items without category/deterministic (IDs 621-622)