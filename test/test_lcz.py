import pytest
import numpy as np
import itertools
import json
import os
import glob
from peony.lcz import inferenceData

def test_inference(tmp_path):
    inferenceData("test/0_0.tif", "test/s2_lcz_weights.hdf5", tmp_path)
    assert os.path.exists(os.path.join(tmp_path, "0_0_lab.tif"))
    assert os.path.exists(os.path.join(tmp_path, "0_0_pro.tif"))
