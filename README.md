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
