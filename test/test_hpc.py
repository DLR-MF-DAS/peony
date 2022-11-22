import pytest
from peony.hpc import pipeline_on_uniform_grid

def test_pipeline_on_uniform_grid(tmp_path):
    pipeline_on_uniform_grid(tmp_path, 'examples.dummy', 5)
