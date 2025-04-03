from typing import List, Optional

class Item:
    def __init__(self, item: str, price: int, quantity: int):
        self._item = item
        self._price = price
        self._quantity = quantity

    def to_dict(self):
        return {
            "item": self._item, 
            "price": self._price,  
            "quantity": self._quantity  
        }


class StoreData:
    def __init__(self, storeName: str, items: List[Item]):
        self._storeName = storeName
        self._items = items if items is not None else []
        
    @property
    def items(self):
        return self._items

    def to_dict(self):
        return {
            "storeName": self._storeName,
            "items": [item.to_dict() for item in self._items] 
        }

class Bill:
    def __init__(self, fileName: str, storeData: StoreData):
        self._fileName = fileName
        self._storeData = storeData

    def to_dict(self):
        return {
            "fileName": self._fileName,
            "storeData": self._storeData.to_dict()  
        }
