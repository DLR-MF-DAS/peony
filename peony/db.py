from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Date
from geoalchemy2 import Geometry

Base = declarative_base()

class Image(Base):
    __tablename__ = 'image'
    id = Column(Integer, primary_key=True)
    path = Column(String)
    geom = Column(Geometry('POLYGON'))
    date = Column(Date)
