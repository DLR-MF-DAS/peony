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

