from peony.inference import bayesian_inference_on_geotiff

def test_bayesian_inference(tmp_path):
    bayesian_inference_on_geotiff("test/Lumberton_ROI_pro.tif", "test/Lumberton_ROI_ESA_WorldCover", tmp_path)
