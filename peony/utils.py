import json
from geoalchemy2 import WKTElement
from pyreproj import Transformer

def geojson_to_wktelement(jsonfile, to_crs='epsg:3857'):
    """Extracts the first polygon from a GeoJSON file.
    
    Parameters
    ----------
    jsonfile: str
        Path to the GeoJSON file.
    
    Returns
    -------
    WKTElement
    """
    with open(jsonfile, 'r') as fd:
        data = json.load(fd)
        coordinates = data["features"][0]["geometry"]["coordinates"][0]
    transformer = Transformer.from_crs('epsg:4326', to_crs)
    coordinates = [transformer.transform(point[0], point[1]) for point in coordinates]
    coordinates = [(point[0] / 10000, point[1] / 10000) for point in coordinates]
    polygon = ", ".join([f"{point[0]} {point[1]}" for point in coordinates])
    return WKTElement(f"Polygon(({polygon}))")
