from typing import Optional
from pydantic import BaseModel

class AssetQuery(BaseModel):
    id: int
    characteristic_form: str
    deterministic: bool
    text: str
    type: Optional[str] = ""  # Optional field with default value

# Required Fields:

  # id: Unique identifier (integer)

  # characteristic_form: Expected answer format (string)

  # deterministic: Boolean flag for answer certainty

  # text: Query content (string)

# Optional Field:

  # type: Empty string by default when not provided