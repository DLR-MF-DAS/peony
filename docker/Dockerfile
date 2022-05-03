FROM ubuntu:20.04

RUN apt-get install software-properties-common && apt-get update && add-apt-repository ppa:ubuntugis/ppa && apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes python3-pip libsqlite3-mod-spatialite default-jre wget git gdal-bin libgdal-dev
RUN pip3 install breathe numpy pandas scikit-learn rasterio matplotlib scikit-image geoalchemy2 spatialite setuptools pytest shapely click jinja2 gdal
RUN pip3 install tensorflow

RUN wget https://step.esa.int/downloads/8.0/installers/esa-snap_sentinel_unix_8_0.sh && yes "" | sh esa-snap_sentinel_unix_8_0.sh

COPY peony/ /peony
RUN cd peony && python3 setup.py install
COPY s2_classification_pipeline
RUN cd s2_classification_pipeline && python3 setup.py install
