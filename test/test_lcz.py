import pytest
import numpy as np
import itertools
import json
import os
import glob
import rasterio
from peony.lcz import inferenceData

def test_inference(tmp_path):
    inferenceData("test/0_0.tif", "test/s2_lcz_weights.hdf5", tmp_path)
    with rasterio.open("0_0_lab_ref.tif") as src:
        lab_ref = src.read()
    with rasterio.open("0_0_pro_ref.tif") as src:
        pro_ref = src.read()  
    assert os.path.exists(os.path.join(tmp_path, "0_0_lab.tif"))
    assert os.path.exists(os.path.join(tmp_path, "0_0_pro.tif"))
    with rasterio.open("0_0_lab.tif") as src:
        lab = src.read()
    with rasterio.open("0_0_pro.tif") as src:
        pro = src.read()
    assert np.all(lab_ref == lab)
    assert np.all(pro_ref == pro)
