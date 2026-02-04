from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_core.core_schema import FieldValidationInfo


class Product(BaseModel):
    product_id: str
    product_name: str
    category: str
    inventory_count: int = Field(ge=0)
    price: float = Field(ge=0)


class Order(BaseModel):
    order_id: int
    customer_id: str
    product_id: str
    quantity: int = Field(ge=0)
    price: float = Field(ge=0)
    order_date: date
    shipping_date: Optional[date] = None
    order_status: str

    @field_validator("shipping_date")
    @classmethod
    def _validate_shipping_date(
        cls, v: Optional[date], info: FieldValidationInfo
    ) -> Optional[date]:
        od = (
            info.data.get("order_date")
            if isinstance(info.data, dict)
            else None
        )
        if v and od and v < od:
            # Do not raise; allow rules layer to record issue instead
            return v
        return v
