FROM ubuntu:20.04

RUN apt-get update && apt-get install --yes software-properties-common && apt-get update && add-apt-repository ppa:ubuntugis/ppa && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install --yes python3-pip libsqlite3-mod-spatialite default-jre wget git gdal-bin libgdal-dev && pip3 install --upgrade pip && pip3 install breathe numpy pandas scikit-learn rasterio matplotlib scikit-image geoalchemy2 spatialite setuptools pytest shapely click jinja2 gdal pypyr s2cloudless && pip3 install tensorflow && wget https://step.esa.int/downloads/8.0/installers/esa-snap_sentinel_unix_8_0.sh && yes "" | sh esa-snap_sentinel_unix_8_0.sh && wget https://step.esa.int/thirdparties/sen2cor/2.10.0/Sen2Cor-02.10.01-Linux64.run && sh Sen2Cor-02.10.01-Linux64.run

COPY . /peony
RUN cd peony && python3 setup.py install
#COPY s2_classification_pipeline/ /s2_classification_pipeline
#RUN cd s2_classification_pipeline && python3 setup.py install
