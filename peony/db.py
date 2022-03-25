from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Date
from sqlalchemy.sql import select, func
from geoalchemy2 import Geometry

Base = declarative_base()

def load_spatialite(dbapi_conn, connection_record):
    dbapi_conn.enable_load_extension(True)
    dbapi_conn.load_extension('mod_spatialite')

def init_spatial_metadata(engine):
    conn = engine.connect()
    conn.execute(select([func.InitSpatialMetaData()]))
    conn.close()

class Image(Base):
    __tablename__ = 'image'
    id = Column(Integer, primary_key=True)
    path = Column(String)
    geom = Column(Geometry('POLYGON', management=True))
    date = Column(Date)
