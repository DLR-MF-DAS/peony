from peony.db import query_polygon
from joblib import Parallel, delayed
from pypyr import pipelinerunner
import os

def pipeline_on_polygon(workdir, pipeline, sqlite_path, polygon, date_range=None, n_jobs=1):
    if not os.path.isdir(workdir):
        raise RuntimeError(f"Work directory {workdir} does not exist!")
    def run_pipeline(path, name):
        subworkdir = os.path.join(workdir, name)
        os.makedirs(subworkdir, exist_ok=True)
        pipelinerunner.run(pipeline_name=pipeline, args_in=[f"path={path}", f"name={name}", f"workdir={subworkdir}", f"logpath={workdir}"])
    entries = query_polygon(sqlite_path, polygon, date_range)
    n_jobs = int(n_jobs)
    Parallel(n_jobs=n_jobs)(delayed(run_pipeline)(entry.path, entry.name) for entry in entries)

