from pydantic import BaseModel
from typing import Optional, List, Union

class BaseQuery(BaseModel):
    id: int
    text: str
    characteristic_form: Optional[str] = None

class IoTQuery(BaseQuery):
    type: str = "IoT"
    category: str

class TSFMQuery(BaseQuery):
    type: str = "TSFM"
    category: str

class WorkorderQuery(BaseQuery):
    type: str = "Workorder"
    deterministic: bool
    note: Optional[str] = None

class AssetQuery(BaseQuery):
    deterministic: bool
    type: Optional[str] = None
    category: Optional[str] = None

class QueryItem(BaseModel):
    __root__: Union[
        IoTQuery,
        TSFMQuery,
        WorkorderQuery,
        AssetQuery
    ]

class Dataset(BaseModel):
    items: List[QueryItem]

# Hierarchical Structure:

    # BaseQuery: Core fields (id, text, characteristic_form)

    # Specialized models for different query types

    # QueryItem as a union type to handle all variations

    # Dataset as the container for all items

# Handles All Variations:

    # IoT queries with type="IoT" and category

    # TSFM queries with type="TSFM" and category

    # Workorder queries with type="Workorder" and deterministic

    # Asset queries with deterministic (and optional type/category)

    # Missing characteristic_form in IDs 621-622

    # Optional note field in workorder queries