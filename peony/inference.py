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

def bayesian_inference_on_geotiff(hypothesis, evidence, likelihood):
    pass
