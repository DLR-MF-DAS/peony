FROM ubuntu:latest

RUN apt-get dist-upgrade
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install --fix-missing --yes python3-pip libsqlite3-mod-spatialite default-jre wget git
RUN pip3 install breathe numpy pandas scikit-learn rasterio matplotlib scikit-image geoalchemy2 spatialite setuptools pytest shapely

RUN wget https://step.esa.int/downloads/8.0/installers/esa-snap_sentinel_unix_8_0.sh && yes "" | sh esa-snap_sentinel_unix_8_0.sh

COPY peony/ /peony
RUN cd peony && python3 setup.py install

