from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.event import listen
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String, Date
from sqlalchemy.sql import select, func
from geoalchemy2 import Geometry, functions
from geoalchemy2.shape import to_shape
from peony.utils import geojson_to_wktelement
import datetime
import pathlib
import json
from os.path import exists
import os
import ee
import pyproj

Base = declarative_base()

class Image(Base):
    """A class representing a single piece of satellite data.
    """
    __tablename__ = 'image'
    
    id = Column(Integer, primary_key=True)
    path = Column(String)
    name = Column(String, unique=True)
    geom = Column(Geometry('POLYGON', management=True))
    date = Column(Date)

def load_spatialite(dbapi_conn, connection_record):
    dbapi_conn.enable_load_extension(True)
    dbapi_conn.load_extension('mod_spatialite')

def init_spatial_metadata(engine):
    conn = engine.connect()
    conn.execute(select([func.InitSpatialMetaData()]))
    conn.close()

def csv_2_spatialite(csv_path, sqlite_path):
    """Populates a spatialite database based on a CSV file.

    Parameters
    ----------
    csv_path: str
        A path to a CSV file with no header and 4 columns.
        The columns are in this order: date, polygon coordinates, 
        name and path.
    sqlite_path: str
        Path to where a spatialite database will be stored.
    """
    engine = create_engine(f"sqlite:///{sqlite_path}")
    listen(engine, 'connect', load_spatialite)
    init_spatial_metadata(engine)
    Image.__table__.create(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    counter = 0
    with open(csv_path, 'r') as fd:
        for line in fd:
            line = line.strip().split(',')
            date_str = line[0].strip('"').split('.')[0]
            date = datetime.datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S')
            polygon = line[1].strip('"').strip().split(' ')
            assert(len(polygon) % 2 == 0)
            assert(polygon[0] == polygon[-2])
            assert(polygon[1] == polygon[-1])
            polygon = ', '.join([polygon[i] + ' ' + polygon[i + 1] for i in range(0, len(polygon), 2)])
            polygon = f"POLYGON(({polygon}))"
            name = line[2].strip('"').strip()
            path_str = line[3].strip('"').strip()
            session.add(Image(path=str(path_str), geom=polygon, name=name, date=date))
            counter += 1
            if (counter % 1000) == 0:
                session.commit()
    session.commit()

def query_polygon(sqlite_path, geojson_path, date_range=None):
    """Will try to find records whos geometry overlaps with the given polygon.

    Parameters
    ----------
    sqlite_path: str
        A path to the sqlite database that contains satellite image metadata.
    geojson_path: str
        A path to a GeoJSON file that contains the polygon to query by.

    Yelds
    -----
    tuple
        A tuple consisting of path, product name and date
    """
    if not exists(sqlite_path):
        raise RuntimeError(f"Database file {sqlite_path} not found!")
    if not exists(geojson_path):
        raise RuntimeError(f"GeoJSON file {geojson_path} with the query polygon not found!")
    engine = create_engine(f"sqlite:///{sqlite_path}")
    listen(engine, 'connect', load_spatialite)
    Session = sessionmaker(bind=engine)
    session = Session()
    polygon = geojson_to_wktelement(geojson_path)
    query = session.query(Image).filter(Image.geom != None).filter(
        Image.geom.ST_Overlaps(polygon))
    if date_range is not None:
        query = query.filter(Image.date >= date_range[0]).filter(Image.date <= date_range[1])
    for image in query:
        yield image

def download_gee_composite(geojson_path, output_path, collection='COPERNICUS/S2_HARMONIZED', mosaic='q-mosaic', cloudless_portion=0.6, max_tile_size=8, start_date="2019-09-01", end_date="2019-12-01", project_name=None, new_algorithm=False):
    """Will download a (hopefully) cloud-free image of a specified region from GEE.

    Parameters
    ----------
    geojson_path: str
        A path to a GeoJSON file that contains the polygon to query by.
    output_path: str
        Name of the output GeoTIFF
    """
    import geedim as gd
    gd.Initialize()
    with open(geojson_path, 'r') as fd:
        data = json.load(fd)
    polygon = data["features"][0]["geometry"]
    basename = os.path.splitext(os.path.basename(geojson_path))[0]
    if new_algorithm:
        utm_code = pyproj.database.query_utm_crs_info(
            datum_name = 'WGS 84',
            area_of_interest = pyproj.aoi.AreaOfInterest(
                west_lon_degree  = polygon["coordinates"][0][0][0],
                south_lat_degree = polygon["coordinates"][0][0][1],
                east_lon_degree  = polygon["coordinates"][0][2][0],
                north_lat_degree = polygon["coordinates"][0][2][1],
            ),
        )[0].code
        try:
            coll = gd.MaskedCollection.from_name(collection)
            coll = coll.search(start_date=start_date, end_date=end_date, region=polygon, cloudless_portion=75, fill_portion=30, custom_filter='CLOUDY_PIXEL_PERCENTAGE<25', prob=40, buffer=100)
            medoid_im = coll.composite('medoid', prob=40, buffer=100)
            medoid_asset_id = f'projects/{project_name}/assets/s2_medoid_{basename}'
            medoid_task = medoid_im.export(medoid_asset_id, type='asset', region=polygon, crs=f"EPSG:{utm_code}", scale=10, dtype='uint16', wait=True)
            medoid_asset_im = gd.MaskedImage.from_id(medoid_asset_id)
            medoid_asset_im.download(output_path, crs=f"EPSG:{utm_code}", scale=10, overwrite=True)
        finally:
            ee.data.deleteAsset(medoid_asset_id)
    else:
        coll = gd.MaskedCollection.from_name(collection)
        coll = coll.search(start_date=start_date, end_date=end_date, region=polygon, cloudless_portion=cloudless_portion)
        comp_im = coll.composite(method=mosaic, region=polygon)
        comp_im.download(output_path, region=polygon, crs="EPSG:4326", scale=10, max_tile_size=max_tile_size)

    
