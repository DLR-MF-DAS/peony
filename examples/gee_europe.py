from peony.hpc import pipeline_on_uniform_grid

if __name__ == '__main__':
    pipeline_on_uniform_grid("/workdir/gee_europe", "gee_lcz", 0.5, longitude_range=[-6.0, 41], latitude_range=[38.0, 66.0], n_jobs=8, overlap=0.05)
