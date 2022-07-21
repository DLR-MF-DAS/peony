
import logging
import os

from pypyr.context import Context
import pypyr.steps.cmd
from pypyr.errors import KeyNotInContextError

def run_step(context: Context) -> None:
    """A custom step that supports dependencies and locking.
    Will not run if inputs are missing, if output is present or if lockfile is present.

    Args:
      context: dict-like. This is the entire pypyr context.
               You can mutate context in this step to make
               keys/values & data available to subsequent
               pipeline steps.

    Returns:
      None.
    """
    logfile = context.get_formatted('logfile')
    stepname = context.get_formatted('stepname')
    path = context.get_formatted('path')
    workdir = context.get_formatted('workdir')
    lockfile = os.path.join(workdir, f"{stepname}.lock")
    try:
        inputfiles = context.get_formatted('inputFiles')
    except KeyNotInContextError:
        inputfiles = []
    try:
        outputfile = context.get_formatted('outputFile')
    except KeyNotInContextError:
        outputfile = None
    # If input files don't exist we skip no matter what
    for inputfile in inputfiles:
        if not os.path.exists(inputfile):
            with open(logfile, 'a') as fd:
                fd.write(f"SKIPPED: step {stepname} skipped in {path} because input file {inputfile} not found\n")
            return
    # If output file exists we skip and delete the lockfile if it exists
    if outputfile is not None:
        if os.path.exists(outputfile):
            with open(logfile, 'a') as fd:
                fd.write(f"SKIPPED: step {stepname} skipped in {path} because output file {outputfile} already present\n")
            try:
                os.remove(lockfile)
            except:
                pass
            return
        else:
            # If output file doesnt exist and there is a lockfile we do nothing
            if os.path.exists(lockfile):
                with open(logfile, 'a') as fd:
                    fd.write(f"SKIPPED: step {stepname} skipped in {path} because output file {outputfile} not yet present and lockfile exists\n")
                return
    # If a lockfile exists we do nothing
    if os.path.exists(lockfile):
        with open(logfile, 'a') as fd:
            fd.write(f"SKIPPED: step {stepname} skipped because lock file found in {path}\n")
        return
    try:
        open(lockfile, 'a').close()
        with open(logfile, 'a') as fd:
            fd.write(f"STARTED: step {stepname} started in {path}\n")
        pypyr.steps.cmd.run_step(context)
    except:
        os.remove(lockfile)
    finally:
        with open(logfile, 'a') as fd:
            fd.write(f"FINISHED: step {stepname} successfully finished in {path}\n")
        try:
            if os.path.exists(outputfile):
                os.remove(lockfile)
        except:
            pass
