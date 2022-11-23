import pytest
import numpy as np
import itertools
import json
import os
import glob
from peony.hpc import pipeline_on_uniform_grid

def test_pipeline_on_uniform_grid(tmp_path):
    grid_size = 5
    workdir = tmp_path
    longitude_range = (-180, 180)
    latitude_range = (-90, 90)
    pipeline_on_uniform_grid(workdir, 'dummy', grid_size)
    nx = int((longitude_range[1] - longitude_range[0]) / grid_size)
    ny = int((latitude_range[1] - latitude_range[0]) / grid_size)
    xs = np.linspace(longitude_range[0], longitude_range[1], nx)
    ys = np.linspace(latitude_range[0], latitude_range[1], ny)
    for i, j in itertools.product(range(nx - 1), range(ny - 1)):
        assert os.path.isdir(os.path.join(workdir, f"{i}_{j}"))
        files = glob.glob(str(os.path.join(workdir, f"{i}_{j}", "*.json")))
        assert(len(files) == 1)
        with open(files[0], 'r') as fd:
            data = json.load(fd)
        assert(len(data["features"][0]["geometry"]["coordinates"][0]) == 5)
