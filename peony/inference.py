import rasterio
from rasterio.enums import Resampling

def bayesian_inference(hypothesis, evidence, likelihood):
    """Apply Bayesian inference given a hypothesis (prior) and evidence (transformed into likelihood).

    Parameters
    ----------
    hypothesis: NumPy array
    evidence: NumPy array
    likelihood: function
    
    Returns
    -------
    NumPy array
        an array with the posterior distribution.
    """
    return likelihood(evidence) * hypothesis

def bayesian_inference_on_geotiff(hypothesis_path, evidence_path, posterior_path, likelihood=lambda x: x):
    with rasterio.open(hypothesis_path) as h_src:
        hypothesis = h_src.read()
        with rasterio.open(evidence_path) as e_src:
            evidence = e_src.read(
                out_shape=(e_src.count, h_src.height, h_src.width),
                resampling=Resampling.nearest)
            e_transform = e_src.transform * e_src.transform.scale(e_src.width / evidence.shape[-1], e_src.height / evidence.shape[-2])
            profile = e_src.profile
            profile['transform'] = e_transform
    posterior = likelihood(evidence) * hypothesis
    with rasterio.open(posterior_path, 'w', **profile) as dst:
        dst.write(posterior)
