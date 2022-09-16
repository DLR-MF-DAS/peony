FROM orbitfold/peony_deps:latest

COPY . /peony
RUN cd peony && conda run -n peony python3 setup.py install
RUN cd peony/s2_classification_pipeline && conda run -n peony python3 setup.py install
