from peony.inference import bayesian_inference_on_geotiff

def test_bayesian_inference(tmp_path):
    bayesian_inference_on_geotiff("test/LumbertonROI_pro.tif", "test/LumbertonROI_ESA_WorldCover.tif", tmp_path)
