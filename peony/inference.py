import rasterio
from rasterio.enums import Resampling
import numpy as np

def bayesian_inference_on_geotiff(hypothesis_path, evidence_path, posterior_path, likelihood=lambda x, y: x, prob_scale=10000):
    with rasterio.open(hypothesis_path) as h_src:
        hypothesis = h_src.read()
        profile = h_src.profile
        with rasterio.open(evidence_path) as e_src:
            evidence = e_src.read(
                out_shape=(e_src.count, h_src.height, h_src.width),
                resampling=Resampling.nearest)
    assert(hypothesis.shape == evidence.shape)
    posterior = likelihood(evidence, hypothesis) * hypothesis
    posterior = posterior / posterior.sum(axis=0).astype(float)
    posterior = np.rint(posterior * prob_scale).astype(int)
    posterior = np.clip(posterior, 0, prob_scale)
    with rasterio.open(posterior_path, 'w', **profile) as dst:
        dst.write(posterior)
