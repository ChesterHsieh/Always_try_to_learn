# infrastructure/orm/models.py

from sqlalchemy import Column, String, Integer, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class ProductModel(Base):
    __tablename__ = 'products'
    id = Column(String, primary_key=True)
    name = Column(String)
    description = Column(String)
    price = Column(Float)
    stock = Column(Integer)

class UserModel(Base):
    __tablename__ = 'users'
    id = Column(String, primary_key=True)
    username = Column(String)
    email = Column(String)
    hashed_password = Column(String)

class OrderModel(Base):
    __tablename__ = 'orders'
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey('users.id'))
    total_amount = Column(Float)
    status = Column(String)
    user = relationship("UserModel", back_populates="orders")

UserModel.orders = relationship("OrderModel", order_by=OrderModel.id, back_populates="user")