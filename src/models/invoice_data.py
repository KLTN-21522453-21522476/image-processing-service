from typing import List, Optional
from pydantic import BaseModel, Field

class Item(BaseModel):
    item: str
    price: int
    quantity: int

class Invoice(BaseModel):
    model: str
    fileName: str
    storeName: str
    items: List[Item] = []
    createdDate: Optional[str] = ""
    id: Optional[str] = ""
    status: Optional[str] = ""
    approvedBy: Optional[str] = ""
    submittedBy: Optional[str] = ""
    address: Optional[str] = ""
    totalAmount: int
    
    class Config:
        # Cho phép sử dụng các tên trường không tuân theo quy tắc đặt tên Python
        populate_by_name = True
        # Cho phép các trường bổ sung không được định nghĩa trong model
        extra = "ignore"
