import json
from geoalchemy2 import WKTElement
from pyreproj import Transformer

def geojson_to_wktelement(jsonfile, to_srs='epsg:3857'):
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
    rp = Reprojector()
    transform = rp.get_transformation_function(from_srs='epsg:4326', to_srs=to_srs)
    coordinates = [transform(point[0], point[1]) for point in coordinates]
    coordinates = [(point[0] / 10000, point[1] / 10000) for point in coordinates]
    polygon = ", ".join([f"{point[0]} {point[1]}" for point in coordinates])
    return WKTElement(f"Polygon(({polygon}))")
