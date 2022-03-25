import pytest
from peony.db import Image, load_spatialite, init_spatial_metadata
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.event import listen
from geoalchemy2 import WKTElement
import datetime

def test_db():
    engine = create_engine('sqlite://', echo=True)
    listen(engine, 'connect', load_spatialite)
    init_spatial_metadata(engine)
    Image.__table__.create(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    session.add(Image(path='/datastore/sentinel/1',
                      geom='POLYGON((0.0 0.0, 0.5 0.0, 0.5 0.5, 0.0 0.5, 0.0 0.0))',
                      date=datetime.date(2019, 1, 1)))
    session.add(Image(path='/datastore/sentinel/2',
                      geom='POLYGON((0.5 0.0, 1.0 0.0, 1.0 0.5, 0.5 0.5, 0.5 0.0))',
                      date=datetime.date(2020, 1, 1)))
    session.add(Image(path='/datastore/sentinel/3',
                      geom='POLYGON((0.0 0.5, 0.5 0.5, 0.5 1.0, 0.0 1.0, 0.0 0.5))',
                      date=datetime.date(2021, 1, 1)))
    session.add(Image(path='/datastore/sentinel/4',
                      geom='POLYGON((0.5 0.5, 1.0 0.5, 1.0 1.0, 0.5 1.0, 0.5 0.5))',
                      date=datetime.date(2022, 1, 1)))
    session.commit()
    query = session.query(Image).filter(
        Image.geom.ST_Overlaps(
            WKTElement(
                'Polygon((0.1 0.1, 0.9 0.1, 0.9 0.4, 0.1 0.4, 0.1 0.1))')))
    results = [image.path for image in query]
    assert(len(results) == 2)
    assert('/datastore/sentinel/1' in results)
    assert('/datastore/sentinel/2' in results)

    
    
