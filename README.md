# Peony

## Installing the Charliecloud Image

The following currently works on LRZ Linux Cluster:

```
$ module load charliecloud
$ mkdir containers
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

## LCZ Example

In this example we will compute LCZ labels for the images from the Sentinel-2 archive that overlap with the Munich area.

```
salloc -pcm2_inter --time=02:00:00 --mem=0 -n 1 srun --pty bash -i
```

```
ch-run -w -b /dss:/dss -b $SCRATCH/workdir:/workdir ~/containers/peony/ -- python3
```

```
import peony.hpc
peony.hpc.pipeline_on_polygon('/workdir', '/peony/examples/lcz_pipeline', '/home/ge83noc2/Sentinel-2_L1C_metadata.sqlite', '/peony/test/munich.json', '10.12.2019-12.12.2019', verbose=True, n_jobs=1)
```

```
context_parser: pypyr.parser.keyvaluepairs
steps:
  - name: peony.cmd
    onError: 'upsampling failed for {path} in {workdir}'
    in:
      stepname: upsample
      outputFile: '{workdir}/upsampled.tif'
      cmd:
        run: /usr/local/snap/bin/gpt /peony/test/bandselect_upsampling.xml -Pinput={path} -Poutput={workdir}/upsampled.tif
        cwd: '{workdir}'
        stdout: '{workdir}/upsample.stdout.log'
        stderr: '{workdir}/upsample.stderr.log'
    
  - name: peony.cmd
    onError: 'inference failed for {path} in {workdir}'
    in:
      stepname: inference
      inputFiles:
        - '{workdir}/upsampled.tif'
      outputFile: '{workdir}/lcz_pro.tif'
      cmd:
        run: sen2inference.py -i {workdir}/upsampled.tif -m /peony/test/s2_lcz_weights.hdf5 -o {workdir}/ --output-file-name=lcz
        cwd: '{workdir}'
        stdout: '{workdir}/inference.stdout.log'
        stderr: '{workdir}/inference.stderr.log'


on_failure:
  - name: pypyr.steps.filewrite
    in:
      fileWrite:
        path: '{logfile}'
        payload: "FAILURE: {runErrors[0][customError]}\n"
        append: True
```

```
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {},
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [
              11.305618286132812,
              48.232906106325146
            ],
            [
              11.241760253906248,
              48.03310084552225
            ],
            [
              11.554183959960938,
              47.94348672179898
            ],
            [
              11.898193359375,
              47.98164918953037
            ],
            [
              11.881027221679688,
              48.21643786815753
            ],
            [
              11.613235473632812,
              48.32749566403233
            ],
            [
              11.305618286132812,
              48.232906106325146
            ]
          ]
        ]
      }
    }
  ]
}
```
