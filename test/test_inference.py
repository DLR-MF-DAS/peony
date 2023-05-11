import os
import numpy as np
from peony.inference import bayesian_inference_on_geotiff
from peony.utils import probability_to_classes, json_to_likelihood
import subprocess
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

def test_bayesian_inference(tmp_path):
    bayesian_inference_on_geotiff("data/Lumberton_ROI_pro.tif", "data/Lumberton_ROI_ESA_WorldCover.tif", os.path.join(tmp_path, 'test.tif'), json_to_likelihood('data/esa_wc_likelihood_uniform.json'))
    with rasterio.open(os.path.join(tmp_path, 'test.tif')) as src:
        data = src.read()
        data = data.astype(float)
        data[data == src.nodata] = np.nan
    assert np.isclose(np.nan_to_num(data.sum(axis=0), nan=10000), 10000, rtol=0, atol=5).all()
    probability_to_classes(os.path.join(tmp_path, 'test.tif'), os.path.join(tmp_path, 'test_lab.tif'), colormap='data/lcz_colormap.json')
    probability_to_classes('data/Lumberton_ROI_pro.tif', os.path.join(tmp_path, 'test_ref_lab.tif'), colormap='data/lcz_colormap.json')
    with rasterio.open(os.path.join(tmp_path, 'test_ref_lab.tif')) as src:
        lab_test_data = src.read()
    with rasterio.open('data/Lumberton_ROI_lab.tif') as src:
        lab_data = src.read()
    assert((lab_test_data == lab_data).all())

def test_bayesian_inference_somalia(tmp_path):
    bayesian_inference_on_geotiff("data/Somalia_pro.tif", "data/Somalia_esa_wc.tif", os.path.join(tmp_path, 'test.tif'), json_to_likelihood('data/esa_wc_likelihood_uniform.json'))
    with rasterio.open(os.path.join(tmp_path, 'test.tif')) as src:
        data = src.read()
        data = data.astype(float)
        data[data == src.nodata] = np.nan
    assert np.isclose(np.nan_to_num(data.sum(axis=0), nan=10000), 10000, rtol=0, atol=5).all()
    probability_to_classes(os.path.join(tmp_path, 'test.tif'), os.path.join(tmp_path, 'test_lab.tif'), colormap='data/lcz_colormap.json')
    probability_to_classes('data/Somalia_pro.tif', os.path.join(tmp_path, 'test_ref_lab.tif'), colormap='data/lcz_colormap.json')
    with rasterio.open(os.path.join(tmp_path, 'test_ref_lab.tif')) as src:
        lab_test_data = src.read()
    with rasterio.open('data/Somalia_lab.tif') as src:
        lab_data = src.read()
    assert(lab_test_data[lab_test_data != lab_data].shape[0] <= 5)

def test_script(tmp_path):
    subprocess.run(['peony_bayesian_inference', '-h', 'data/Lumberton_ROI_pro.tif', '-e', 'data/Lumberton_ROI_ESA_WorldCover.tif', '-p', os.path.join(tmp_path, 'test_pro.tif'), '-l', 'data/esa_wc_likelihood_uniform.json'])
    assert(os.path.exists(os.path.join(tmp_path, 'test_pro.tif')))
    subprocess.run(['peony_pro_to_lab', '-i', os.path.join(tmp_path, 'test_pro.tif'), '-o', os.path.join(tmp_path, 'test_lab.tif')])
    assert(os.path.exists(os.path.join(tmp_path, 'test_lab.tif')))
    subprocess.run(['peony_bayesian_inference', '-h', 'data/Somalia_pro.tif', '-e', 'data/Somalia_esa_wc.tif', '-p', os.path.join(tmp_path, 'somalia_pro.tif'), '-l', 'data/esa_wc_likelihood_uniform.json'])
    assert(os.path.exists(os.path.join(tmp_path, 'somalia_pro.tif')))
    subprocess.run(['peony_pro_to_lab', '-i', os.path.join(tmp_path, 'somalia_pro.tif'), '-o', os.path.join(tmp_path, 'somalia_lab.tif')])
    assert(os.path.exists(os.path.join(tmp_path, 'somalia_lab.tif')))
