import os
import numpy as np
from peony.inference import bayesian_inference_on_geotiff
from peony.utils import probability_to_classes
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

def esa_world_cover_to_lcz_likelihood(esa_wc, lcz):
    likelihood = np.zeros(lcz.shape)
    esa_wc = esa_wc[0]
    assert esa_wc.shape == likelihood.shape[1:], f"{esa_wc.shape} != {likelihood.shape}"
    p = [
        np.nonzero(esa_wc == 10),
        np.nonzero(esa_wc == 20),
        np.nonzero(esa_wc == 30),
        np.nonzero(esa_wc == 40),
        np.nonzero(esa_wc == 50),
        np.nonzero(esa_wc == 60),
        np.nonzero(esa_wc == 70),
        np.nonzero(esa_wc == 80),
        np.nonzero(esa_wc == 90),
        np.nonzero(esa_wc == 95),
        np.nonzero(esa_wc == 100)
    ]
    likelihood[:, p[0][0], p[0][1]] = np.transpose(np.repeat(np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1]]), p[0][0].shape[0], axis=0))
    likelihood[:, p[1][0], p[1][1]] = np.transpose(np.repeat(np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.9, 0.5, 0.1, 0.1, 0.1]]), p[1][0].shape[0], axis=0))
    likelihood[:, p[2][0], p[2][1]] = np.transpose(np.repeat(np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1]]), p[2][0].shape[0], axis=0))
    likelihood[:, p[3][0], p[3][1]] = np.transpose(np.repeat(np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1]]), p[3][0].shape[0], axis=0))
    likelihood[:, p[4][0], p[4][1]] = np.transpose(np.repeat(np.array([[0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]), p[4][0].shape[0], axis=0))
    likelihood[:, p[5][0], p[5][1]] = np.transpose(np.repeat(np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.1, 0.1, 0.1]]), p[5][0].shape[0], axis=0))
    likelihood[:, p[6][0], p[6][1]] = np.transpose(np.repeat(np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]), p[6][0].shape[0], axis=0))
    likelihood[:, p[7][0], p[7][1]] = np.transpose(np.repeat(np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9]]), p[7][0].shape[0], axis=0))
    likelihood[:, p[8][0], p[8][1]] = np.transpose(np.repeat(np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]), p[8][0].shape[0], axis=0))
    likelihood[:, p[9][0], p[9][1]] = np.transpose(np.repeat(np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]), p[9][0].shape[0], axis=0))
    likelihood[:, p[10][0], p[10][1]] = np.transpose(np.repeat(np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]), p[10][0].shape[0], axis=0))
    return likelihood

def test_bayesian_inference(tmp_path):
    bayesian_inference_on_geotiff("test/Lumberton_ROI_pro.tif", "test/Lumberton_ROI_ESA_WorldCover.tif", os.path.join(tmp_path, 'test.tif'), esa_world_cover_to_lcz_likelihood)
    with rasterio.open(os.path.join(tmp_path, 'test.tif')) as src:
        data = src.read()
        data = data.astype(float)
        data[data == src.nodata] = np.nan
    assert np.isclose(np.nan_to_num(data.sum(axis=0), nan=10000), 10000, rtol=0, atol=5).all()
    probability_to_classes(os.path.join(tmp_path, 'test.tif'), os.path.join(tmp_path, 'test_lab.tif'), colormap='data/lcz_colormap.json')
    probability_to_classes('test/Lumberton_ROI_pro.tif', os.path.join(tmp_path, 'test_ref_lab.tif'), colormap='data/lcz_colormap.json')
    with rasterio.open(os.path.join(tmp_path, 'test_ref_lab.tif')) as src:
        lab_test_data = src.read()
    with rasterio.open('test/Lumberton_ROI_lab.tif') as src:
        lab_data = src.read()
    assert((lab_test_data == lab_data).all())

def test_bayesian_inference_somalia(tmp_path):
    bayesian_inference_on_geotiff("data/Somalia_pro.tif", "data/Somalia_esa_wc.tif", os.path.join(tmp_path, 'test.tif'), esa_world_cover_to_lcz_likelihood)
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
    subprocess.run(['peony_bayesian_inference', '-h', 'test/Lumberton_ROI_pro.tif', '-e', 'test/Lumberton_ROI_ESA_WorldCover.tif', '-p', os.path.join(tmp_path, 'test_pro.tif'), '-l', 'data/esa_wc_likelihood_uniform.json'])
    assert(os.path.exists(os.path.join(tmp_path, 'test_pro.tif')))
    subprocess.run(['peony_pro_to_lab', '-i', os.path.join(tmp_path, 'test_pro.tif'), '-o', os.path.join(tmp_path, 'test_lab.tif')])
    assert(os.path.exists(os.path.join(tmp_path, 'test_lab.tif')))
    with rasterio.open('test/Lumberton_ROI_lab.tif') as src:
        data_1 = src.read()
    with rasterio.open(os.path.join(tmp_path, 'test_lab.tif')) as src:
        data_2 = src.read()
    assert((data_1 == data_2).all())
    subprocess.run(['peony_bayesian_inference', '-h', 'data/Somalia_pro.tif', '-e', 'data/Somalia_esa_wc.tif', '-p', os.path.join(tmp_path, 'somalia_pro.tif'), '-l', 'data/esa_wc_likelihood_uniform.json'])
    assert(os.path.exists(os.path.join(tmp_path, 'somalia_pro.tif')))
    subprocess.run(['peony_pro_to_lab', '-i', os.path.join(tmp_path, 'somalia_pro.tif'), '-o', os.path.join(tmp_path, 'somalia_lab.tif')])
    assert(os.path.exists(os.path.join(tmp_path, 'somalia_lab.tif')))
