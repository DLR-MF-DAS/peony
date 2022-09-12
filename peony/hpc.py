from peony.db import query_polygon
from joblib import Parallel, delayed
from pypyr import pipelinerunner
import logging
import os

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

