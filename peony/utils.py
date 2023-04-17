import json
from geoalchemy2 import WKTElement

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
    polygon = ", ".join([f"{point[1]} {point[0]}" for point in coordinates])
    return WKTElement(f"Polygon(({polygon}))")

def probability_to_classes(pro_geotiff, lab_geotiff):
    """Reads a geotiff with probability distribution and write a corresponding label file.
    Using max likelihood.
    """
    pass
