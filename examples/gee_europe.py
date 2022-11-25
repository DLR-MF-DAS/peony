from peony.hpc import pipeline_on_uniform_grid

if __name__ == '__main__':
    pipeline_on_uniform_grid("/workdir/gee_europe", "gee_lcz", 0.5, longitude_range=[-27.0, 45], latitude_range=[35.0, 65.0], n_jobs=5)
