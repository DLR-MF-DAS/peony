import os
import numpy as np
from peony.inference import bayesian_inference_on_geotiff
from peony.utils import probability_to_classes
import rasterio

# ESA WorldCover classes
# ----------------------
# 10 - Tree cover 
# 20 - Shrubland
# 30 - Grassland
# 40 - Cropland
# 50 - Built-up
# 60 - Bare/sparse vegetation
# 70 - Snow and ice 
# 80 - Permanent water bodies
# 90 - Herbaceous wetland
# 95 - Mangroves
# 100 - Moss and lichen

# LCZ classes
# -----------
# 1 - Compact Highrise
# 2 - Compact Midrise
# 3 - Compact Lowrise
# 4 - Open Highrise
# 5 - Open Midrise
# 6 - Open Lowrise
# 7 - Lightweight Lowrise
# 8 - Large Lowrise
# 9 - Sparsely Built
# 10 - Heavy Industry
# 11 (A) - Dense Trees
# 12 (B) - Scattered Trees
# 13 (C) - Bush, Scrub
# 14 (D) - Low Plants
# 15 (E) - Bare Rock/Paved
# 16 (F) - Bare Soil/Sand
# 17 (G) - Water

def esa_world_cover_to_lcz_likelihood(esa_wc, lcz):
    likelihood = np.zeros(lcz.shape)
    likelihood = np.swapaxes(likelihood, 0, 2)
    likelihood = np.swapaxes(likelihood, 0, 1)
    esa_wc = esa_wc[0]
    likelihood[np.argwhere(esa_wc == 10)] = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
    likelihood[np.argwhere(esa_wc == 20)] = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.9, 0.5, 0.1, 0.1, 0.1])
    likelihood[np.argwhere(esa_wc == 30)] = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1])
    likelihood[np.argwhere(esa_wc == 40)] = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1])
    likelihood[np.argwhere(esa_wc == 50)] = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    likelihood[np.argwhere(esa_wc == 60)] = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.1, 0.1, 0.1])
    likelihood[np.argwhere(esa_wc == 70)] = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    likelihood[np.argwhere(esa_wc == 80)] = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9])
    likelihood[np.argwhere(esa_wc == 90)] = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    likelihood[np.argwhere(esa_wc == 95)] = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    likelihood[np.argwhere(esa_wc == 100)] = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    likelihood = np.swapaxes(likelihood, 0, 1)
    likelihood = np.swapaxes(likelihood, 0, 2)
    return likelihood

def test_bayesian_inference(tmp_path):
    bayesian_inference_on_geotiff("test/Lumberton_ROI_pro.tif", "test/Lumberton_ROI_ESA_WorldCover.tif", os.path.join(tmp_path, 'test.tif'), esa_world_cover_to_lcz_likelihood)
    with rasterio.open(os.path.join(tmp_path, 'test.tif')) as src:
        data = src.read()
    assert(np.isclose(data.sum(axis=0), 10000, rtol=0, atol=5).all())
    probability_to_classes(os.path.join(tmp_path, 'test.tif'), os.path.join(tmp_path, 'test_lab.tif'))
    with rasterio.open(os.path.join(tmp_path, 'test_lab.tif')):
        lab_test_data = src.read()
    with rasterio.open('test/Lumberton_ROI_lab.tif') as src:
        lab_data = src.read()
    assert((lab_test_data == lab_data).all())
