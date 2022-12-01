import pytest
import numpy as np
import itertools
import json
import os
import glob
from peony.lcz import inferenceData

def test_inference(tmp_path):
    inferenceData("test/barcelona_mosaic.tiff", "test/s2_lcz_weights.hdf5", tmp_path)
    assert os.path.exists(os.path.join(tmp_path, "barcelona_mosaic_lab.tiff"))
    assert os.path.exists(os.path.join(tmp_path, "barcelona_mosaic_prob.tiff"))