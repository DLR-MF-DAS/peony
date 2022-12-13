from peony.db import query_polygon
import itertools
import tempfile
import numpy as np
from joblib import Parallel, delayed
from pypyr import pipelinerunner
import logging
import os
import json
import glob

def pipeline_on_polygon(workdir, pipeline, sqlite_path, polygon, date_range=None, n_jobs=1, verbose=False):
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    if date_range is not None:
        import datetime
        date_pair = date_range.strip().split('-')
        date_pair = (datetime.datetime.strptime(date_pair[0], '%d.%m.%Y'),
                     datetime.datetime.strptime(date_pair[1], '%d.%m.%Y'))
    else:
        date_pair = None
    if not os.path.isdir(workdir):
        raise RuntimeError(f"Work directory {workdir} does not exist!")
    def run_pipeline(path, date, name):
        subworkdir = os.path.join(workdir, str(date), name)
        os.makedirs(subworkdir, exist_ok=True)
        pipelinerunner.run(pipeline_name=pipeline, args_in=[f"path={path}", f"name={name}", f"workdir={subworkdir}", f"logfile={workdir}/logfile.log"])
    entries = query_polygon(sqlite_path, polygon, date_pair)
    n_jobs = int(n_jobs)
    Parallel(n_jobs=n_jobs)(delayed(run_pipeline)(entry.path, entry.date, entry.name) for entry in entries)

def pipeline_on_uniform_grid(workdir, pipeline, grid_size, longitude_range=(-180, 180), latitude_range=(-90, 90), n_jobs=1, overlap_percentage=0.0):
    assert(longitude_range[0] < longitude_range[1])
    assert(latitude_range[0] < latitude_range[1])
    nx = int((longitude_range[1] - longitude_range[0]) / grid_size)
    ny = int((latitude_range[1] - latitude_range[0]) / grid_size)
    xs = np.linspace(longitude_range[0], longitude_range[1], nx)
    ys = np.linspace(latitude_range[0], latitude_range[1], ny)
    overlap = (overlap_percentage * grid_size) * 0.5
    def run_pipeline(i, j):
        rectangle = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "coordinates": [[(xs[i] - overlap, ys[j] - overlap), (xs[i + 1] + overlap, ys[j] - overlap), (xs[i + 1] + overlap, ys[j + 1] + overlap), (xs[i] - overlap, ys[j + 1] + overlap), (xs[i] - overlap, ys[j] - overlap)]],
                "type": "Polygon"
                }
            }]
        }
        subworkdir = os.path.join(workdir, f"{i}_{j}")
        os.makedirs(subworkdir, exist_ok=True)
        filename = os.path.join(subworkdir, f"{i}_{j}.json")
        if not os.path.exists(filename):
            with open(filename, 'w') as fd:
                json.dump(rectangle, fd)
        pipelinerunner.run(pipeline_name=pipeline, args_in=[f"name={i}_{j}", f"path={filename}", f"workdir={subworkdir}", f"logfile={workdir}/logfile.log"])
    info = {'pipeline' : pipeline, 'grid_size' : grid_size, 'longitude_range' : list(longitude_range_, 'latitude_range' : list(latitude_range)}
    with open('info.json', 'w') as fd:
        json.dump(info, fd)
    Parallel(n_jobs=n_jobs)(delayed(run_pipeline)(i, j) for i, j in itertools.product(range(nx - 1), range(ny - 1)))

def grid_progress(workdir):
    pass
