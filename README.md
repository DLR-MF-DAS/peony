# Peony

## Installing the Charliecloud Image

The following currently works on LRZ Linux Cluster:

```
$ module load charliecloud$ mkdir containers
$ mkdir $SCRATCH/image_storage
$ export CH_IMAGE_STORAGE=$SCRATCH/image_storage
$ git clone https://gitlab.dlr.de/janc_vy/peony.git
$ cd peony
$ git submodule update --init
$ ch-build2dir --force -t peony -f docker/Dockerfile . ../containers
```

The building of the image and the container will take a considerable amount of time in my experience. It will be faster if updating however. It should help to request a full node for interactive use instead of using the login node.
Charliecloud allows one to build an image without any special privileges.

## Running the Charliecloud Container

A simple way to test if the installation succeeded is to run the following command:

```
$ ch-run -w -b /dss:/dss ~/containers/peony/ -- peony -p /peony/test/munich.json -d /home/$USER/Sentinel-2_L1C_metadata.sqlite —date-range=14.06.2019-15.06.2019
```

A couple of notes on what is going on:
  * The command ch-run is used to run applications inside a Charliecloud container.
  * The option -b /dss:/dss will mount the image archive in the same place in the container as it appears for the users of the Linux Cluster/SuperMUC-NG.       Charliecloud will automatically mount the home directory under /home/$USER.
  * The GeoJSON file munich.json I have created in advance.
