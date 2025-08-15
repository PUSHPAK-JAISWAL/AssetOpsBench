from pydantic import BaseModel

class IoTQuery(BaseModel):
    id: int
    type: str
    text: str
    category: str
    characteristic_form: str


# id as integer

# type and category as strings (e.g., "IoT", "Knowledge Query")

# text containing the original query

# characteristic_form describing expected response format