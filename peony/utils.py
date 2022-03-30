import json
from geoalchemy2 import WKTElement

def geojson_to_wktelement(jsonfile):
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
    polygon = ", ".join([f"{point[0]} {point[1]}" for point in coordinates])
    return WKTElement(f"Polygon(({polygon}))")
