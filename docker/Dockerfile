FROM orbitfold/peony_deps:latest

COPY . /peony
RUN cd peony && python3 setup.py install
RUN cd peony/s2_classification_pipeline && python3 setup.py install
