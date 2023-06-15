import json
import rasterio
import numpy as np
from geoalchemy2 import WKTElement
from scipy.interpolate import RegularGridInterpolator
import logging
from ast import literal_eval


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


def probability_to_classes(pro_geotiff, lab_geotiff, index_to_label=lambda x: x + 1, colormap=None):
    """Reads a geotiff with probability distribution and write a corresponding label file.
    Using max likelihood.
    """
    with rasterio.open(pro_geotiff) as src:
        pro = src.read()
        profile = src.profile
    lab = index_to_label(np.argmax(pro, axis=0))
    profile.update(dtype=rasterio.uint8, count=1, compress='lzw', nodata=0)
    with rasterio.open(lab_geotiff, 'w', **profile) as dst:
        dst.write(lab, 1)
        if colormap is not None:
            with open(colormap, 'r') as fd:
                cmap = json.load(fd)
                cmap = {int(k): v for k, v in cmap.items()}
            dst.write_colormap(1, cmap)

def dict_to_normalized_list(d):
    """A utility function to convert dictionary to a normalized (adds up to 1) list"""
    l = list(d.values())
    s = sum(l)
    if s > 0.0:
        l = [v / s for v in l]
    else:
        l = [1.0 / float(len(l)) for v in l]
    return l

def json_to_likelihood(json_file, nodata=None):
    """Create a likelihood from a json file.

    Parameters
    ----------
    json_file: str
      A json file with a likelihood description

    Returns
    -------
    A likelihood function to be used with bayesian inference functions
    """
    with open(json_file, 'r') as fd:
        data = json.load(fd)
    def likelihood_function(evidence, hypothesis):
        likelihood = np.zeros(hypothesis.shape)
        cumulative_matches = np.zeros(evidence.shape)
        cumulative_matches = cumulative_matches.astype(bool)
        for key in data:
            if key not in ['nodata', 'otherwise']:
                eval_key = literal_eval(key)
                if isinstance(eval_key, tuple):
                    matches = np.nonzero((eval_key[0] <= evidence) & (evidence < eval_key[1]))
                    cumulative_matches += ((eval_key[0] <= evidence) & (evidence < eval_key[1]))
                else:
                    matches = np.nonzero(evidence == eval_key)
                    cumulative_matches += (evidence == eval_key)
                likelihood[:, matches[0], matches[1]] = np.transpose(np.repeat(np.array([dict_to_normalized_list(data[key])]), matches[0].shape[0], axis=0))
            try:
                matches = np.nonzero(evidence == nodata)
                cumulative_matches += (evidence == nodata)
                likelihood[:, matches[0], matches[1]] = np.transpose(np.repeat(np.array([dict_to_normalized_list(data['nodata'])]), matches[0].shape[0], axis=0))
            except KeyError:
                logging.debug('no nodata likelihood specified')
            try:
                matches = np.nonzero(np.logical_not(cumulative_matches))
                likelihood[:, matches[0], matches[1]] = np.transpose(np.repeat(np.array([dict_to_normalized_list(data['otherwise'])]), matches[0].shape[0], axis=0))
            except KeyError:
                logging.debug('no otherwise likelihood specified')
        return likelihood
    return likelihood_function


def resample_2d(arr, h, w, method='nearest'):
    """Resample a 2D array to new size using nearest neighbor interpolation.

    Parameters
    ----------
    arr: NumPy array
      An arbitrary 2D array
    h: float
      New height
    w: float
      New width
    method: str
      Name of the interpolation method

    Returns
    -------
    NumPy array
      Resampled array
    """
    assert(len(arr.shape) == 2)
    interp = RegularGridInterpolator((np.arange(arr.shape[0]), np.arange(arr.shape[1])), arr, method=method)
    new_x = np.linspace(0, arr.shape[0] - 1, h)
    new_y = np.linspace(0, arr.shape[1] - 1, w)
    new_xg, new_yg = np.meshgrid(new_x, new_y, indexing='ij')
    result = interp((new_xg, new_yg))
    return result
