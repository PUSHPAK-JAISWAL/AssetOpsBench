from pydantic import BaseModel

class TSFMQuery(BaseModel):
    id: int
    type: str
    text: str
    category: str
    characteristic_form: str

# id as integer identifier  

# type always "TSFM" in your examples

# text containing the original question

# category for query type ("Knowledge Query", "Inference Query", etc.)

# haracteristic_form specifying expected response format