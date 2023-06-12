import rasterio
from rasterio.enums import Resampling
import numpy as np
from peony.utils import resample_2d

def bayesian_inference_on_geotiff(hypothesis_path, evidence_path, posterior_path, likelihood=lambda x, y: x, prob_scale=10000, band=0):
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
    #assert hypothesis.shape[1:] == evidence.shape, f"hypothesis shape {hypothesis.shape} is not the same as evidence shape {evidence.shape}"
    posterior = likelihood(evidence, hypothesis) * hypothesis
    posterior = posterior / posterior.sum(axis=0).astype(float)
    assert np.isclose(np.nan_to_num(posterior.sum(axis=0), nan=1.0), 1.0).all()
    with rasterio.open(posterior_path, 'w', **profile) as dst:
        posterior = np.rint(posterior * prob_scale).astype(int)
        posterior = np.clip(posterior, 0, prob_scale)
        dst.write(posterior)

def likelihood_from_confusion_matrix(confusion_matrix, mapping):
    return {}

