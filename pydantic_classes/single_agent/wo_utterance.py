from pydantic import BaseModel
from typing import Optional

class WorkorderQuery(BaseModel):
    id: int
    text: str
    type: str
    deterministic: bool
    characteristic_form: str
    note: Optional[str] = None  # Optional field for additional notes

# Required Fields:

#     id: Unique integer identifier

#     text: The actual query text

#     type: Always "Workorder" in your dataset

#     deterministic: Boolean flag indicating if the query has a deterministic answer

#     characteristic_form: Description of the expected response format

# Optional Field:

#     note: Only present in some entries (e.g., ID 409), defaults to None