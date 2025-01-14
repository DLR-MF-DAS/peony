import rasterio
import logging
from rasterio.enums import Resampling
import numpy as np
from peony.utils import resample_2d, add_suffix_to_filename
import itertools
from ast import literal_eval
import json

DISTRIBUTIONS = {
    "gaussian": lambda mu, sigma: lambda x: (1.0 / np.sqrt(2.0 * np.pi * sigma ** 2)) * np.exp(-(x - mu) ** 2 / (2.0 * sigma ** 2)),
    "uniform": lambda a, b: lambda x: 1.0 / (b - a),
    "constant": lambda c: lambda x: c
}


class Likelihood:
    def __init__(self, json_file, mapping=None, nodata=None):
        if mapping is None:
            with open(json_file, 'r') as fd:
                self.data = json.load(fd)
        else:
            with open(json_file, 'r') as fd:
                c = json.load(fd)
            with open(mapping, 'r') as fd:
                m = json.load(fd)
            self.data = self._from_confusion_matrix(c, m)
        self.nodata = nodata

    def _from_confusion_matrix(self, confusion, mapping):
        data = {}
        for e_key in confusion:
            data[e_key] = {}
            for h_key in mapping:
                data[e_key][h_key] = 0.0
        for e_key in confusion:
            for h_key in mapping:
                for m_key in mapping[h_key]:
                    data[e_key][h_key] += confusion[m_key][e_key]
        for e_key in data:
            for h_key in data[e_key]:
                data[e_key][h_key] = {"type": "constant", "params": {"c": data[e_key][h_key]}}
        return data

    def __call__(self, evidence, hypothesis):
        assert(evidence.shape == hypothesis.shape[1:])
        likelihood = np.full(hypothesis.shape, None)
        for key in self.data:
            for i, j in itertools.product(range(evidence.shape[0]), range(evidence.shape[1])):
                e = evidence[i][j]
                for i_h, h in enumerate(self.data[key]):
                    pdf = DISTRIBUTIONS[self.data[key][h]["type"]](**self.data[key][h]["params"])
                    if key not in ["nodata", "otherwise"]:
                        likelihood[i_h][i][j] = pdf(e)
                    elif key == "nodata":
                        if e == self.nodata:
                            likelihood[i_h][i][j] = pdf(e)
        try:
            for i, j in itertools.product(range(evidence.shape[0]), range(evidence.shape[1])):
                e = evidence[i][j]
                for i_h, h in enumerate(self.data["otherwise"]):
                    if likelihood[i_h][i][j] is None:
                        pdf = DISTRIBUTIONS[self.data["otherwise"][h]["type"]](**self.data["otherwise"][h]["params"])
                        likelihood[i_h][i][j] = pdf(e)
        except KeyError:
            logging.debug("No otherwise key specified in the likelihood description")
        likelihood = likelihood.astype(float)
        likelihood = likelihood / likelihood.sum(axis=0)
        return likelihood


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


def bayesian_inference_on_geotiff(hypothesis_path, evidence_path, posterior_path, likelihood=lambda x, y: x, prob_scale=10000, band=0):
    """Perform Bayesian inference on two GeoTIFFs (one with a prior probability and one with evidence).
    """
    with rasterio.open(hypothesis_path) as h_src:
        hypothesis = h_src.read()
        profile = h_src.profile
        with rasterio.open(evidence_path) as e_src:
            evidence = e_src.read()
            if band is not None:
                evidence = resample_2d(evidence[band], h_src.height, h_src.width)
            else:
                for band in range(evidence.shape[0]):
                    evidence[band] = resample_2d(evidence[band], h_src.height, h_src.width)
        resampled_profile = profile.copy()
        resampled_profile.update({"count": 1})
        with rasterio.open(add_suffix_to_filename(evidence_path, "_resampled.tiff"), 'w', **resampled_profile) as dst:
            dst.write(evidence, 1)
    posterior = likelihood(evidence, hypothesis) * hypothesis
    posterior = posterior / posterior.sum(axis=0).astype(float)
    assert np.isclose(np.nan_to_num(posterior.sum(axis=0), nan=1.0), 1.0).all()
    with rasterio.open(posterior_path, 'w', **profile) as dst:
        posterior = np.rint(posterior * prob_scale).astype(int)
        posterior = np.clip(posterior, 0, prob_scale)
        dst.write(posterior)


def likelihood_from_confusion_matrix(confusion, mapping):
    likelihood = {}
    for e_key in confusion:
        likelihood[e_key] = {}
        for h_key in mapping:
            likelihood[e_key][h_key] = 0.0
    for e_key in confusion:
        for h_key in mapping:
            for m_key in mapping[h_key]:
                likelihood[e_key][h_key] += confusion[m_key][e_key]
    return likelihood

