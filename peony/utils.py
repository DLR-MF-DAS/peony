import json
import rasterio
import numpy as np
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

def probability_to_classes(pro_geotiff, lab_geotiff, index_to_label=lambda x: x + 1):
    """Reads a geotiff with probability distribution and write a corresponding label file.
    Using max likelihood.
    """
    with rasterio.open(pro_geotiff) as src:
        pro = src.read()
        profile = src.profile
    lab = index_to_label(np.argmax(pro, axis=0))
    profile.update(dtype=rasterio.uint8, count=1, compress='lzw')
    with rasterio.open(lab_geotiff, 'w', **profile) as dst:
        dst.write(lab, 1)
