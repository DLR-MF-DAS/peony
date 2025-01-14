import pytest
import numpy as np
import itertools
import json
import os
import glob
import rasterio
from peony.lcz import inferenceData

def test_inference(tmp_path):
    inferenceData("data/0_0.tif", "data/s2_lcz_weights.hdf5", tmp_path)
    with rasterio.open("data/0_0_lab_ref.tif") as src:
        lab_ref = src.read()
    with rasterio.open("data/0_0_pro_ref.tif") as src:
        pro_ref = src.read()  
    assert os.path.exists(os.path.join(tmp_path, "0_0_lab.tif"))
    assert os.path.exists(os.path.join(tmp_path, "0_0_pro.tif"))
    with rasterio.open(os.path.join(tmp_path, "0_0_lab.tif")) as src:
        lab = src.read()
    with rasterio.open(os.path.join(tmp_path, "0_0_pro.tif")) as src:
        pro = src.read()
    assert np.all(lab_ref == lab)
    #assert np.array_equal(pro_ref, pro)
