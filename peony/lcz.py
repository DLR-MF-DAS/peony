# based on code by Nikolai Skuppin

import os
import sys
import numpy as np
from pathlib import Path

from osgeo import gdal
from osgeo import gdal_array

gdal.UseExceptions()

import tensorflow as tf
from tensorflow.keras import mixed_precision

from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Reshape, Activation, Permute, Lambda, Concatenate,GlobalAveragePooling2D
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def resnet_v2(input_shape, depth, num_classes=10, final_activation='softmax'):
    """ResNet Version 2 Model builder [b]
    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
        final_activation (str): activation applied to last dense layer: put 'softmax' or None
                                'softmax' for probabilities and None for logits
    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
                
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    kernel_initializer='he_normal')(y)
    if final_activation is not None:
        outputs = Activation(final_activation, dtype='float32')(outputs)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def writeGeoTiff(filename, data, proj, geoInfo, type=np.int16):
    """Write data to filename as GeoTiff.
    
    Parameters:
    -----------
    filename : str
        Path to store file.

    data : np.array((nBand, x, y))
        Data to be stored (nBand images of size x, y)

    proj : projection
        Projection to be used for output file.

    geoInfo : geo transform
        Geo transform to be used for outout file.

    type : np.dtype
        Data type to be used for storing (auto covnverted to GDAL type).  
    """
    bnd = data.shape[0]
    row = data.shape[1]
    col = data.shape[2]

    GDALDriver = gdal.GetDriverByName('GTiff')
    File = GDALDriver.Create(filename, col, row, bnd,
                    gdal_array.NumericTypeCodeToGDALTypeCode(type))
    File.SetProjection(proj)
    File.SetGeoTransform(geoInfo)

    # save file with int zeros
    idBnd = np.arange(0, bnd, dtype=int)
    for idxBnd in idBnd:
        outBand = File.GetRasterBand(int(idxBnd+1))
        outBand.WriteArray(data[idxBnd,:,:].astype(type))
        outBand.FlushCache()
        del(outBand)
    File = None


def applyLCZColors(band):
    """Apply LCZ color scheme to rater band.
    
    Parameters:
    -----------
    band : GDAL RasterBand
        Band to apply color scheme to

    Returns
    -------
    GDAL RasterBand 
        Band with LCZ color scheme apllied to
    """
    # create color table
    colors = gdal.ColorTable()

    # set color for each value
    colors.SetColorEntry(1,  (165,   0,  33))
    colors.SetColorEntry(2,  (204,   0,   0))
    colors.SetColorEntry(3,  (255,   0,   0))
    colors.SetColorEntry(4,  (153,  51,   0))
    colors.SetColorEntry(5,  (204, 102,   0))
    colors.SetColorEntry(6,  (255, 153,   0))
    colors.SetColorEntry(7,  (255, 255,   0))
    colors.SetColorEntry(8,  (192, 192, 192))
    colors.SetColorEntry(9,  (255, 204, 153))
    colors.SetColorEntry(10, ( 77,  77,  77))

    colors.SetColorEntry(11, (  0, 102,   0))
    colors.SetColorEntry(12, ( 21, 255,  21))
    colors.SetColorEntry(13, (102, 153,   0))
    colors.SetColorEntry(14, (204, 255, 102))
    colors.SetColorEntry(15, (  0,   0, 102))
    colors.SetColorEntry(16, (255, 255, 204))
    colors.SetColorEntry(17, ( 51, 102, 255))

    # set color table and color interpretation
    band.SetRasterColorTable(colors)
    band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
    return band


class GDALHelper():
    """Helper class to handle GeoTiff files and write results to GEOTiff.
        data : np.array((nBands, rows, columns)) 
            Image to be processed.
    """
    def __init__(self, filename, readData=False, bands=None, scale=1):
        try:
            fid = gdal.Open(filename)
            self.row = fid.RasterYSize
            self.col = fid.RasterXSize
            self.bnd = fid.RasterCount
            self.proj = fid.GetProjection()
            self.geoInfo = fid.GetGeoTransform()
            if readData is True:
                if bands is None:
                    print('Missing band selection in GDALHelper. Please specify bands.')
                for ind, band in enumerate(bands):
                    srcband = fid.GetRasterBand(band)

                    if srcband is None:
                        print('srcband is None' + str(band) + filename)
                        continue
                    arr = srcband.ReadAsArray()

                    if ind==0:
                        R = arr.shape[0]
                        C = arr.shape[1]
                        self.data = np.zeros((len(bands), R, C), dtype=np.float32)

                    self.data[ind,:,:]=np.float32(arr)/scale

        except RuntimeError as e:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("ERROR:           the given data geotiff can not be open by GDAL")
            print("DIRECTORY:       " + filename)
            print("GDAL EXCEPCTION: " + e)
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            sys.exit(1)

    def getCoordLCZGrid(self):
        """this function gets the coordinate of every cell for the LCZ classification map.
        Returns
        -------
        np.array() 
            The coordinate of each cell of the map grid. 
            A N by 2 array with N is the number of cell, 1st col is x-coordinate, 
            2nd col is y-coordinate, The coordinate organized line by line.
        """
        # read the grid and find coordinate of each cell
        row_cell = np.arange(0, self.row)
        col_cell = np.arange(0, self.col)

        geoInfoGrid = self.geoInfo

        xWorld = geoInfoGrid[0] + col_cell * geoInfoGrid[1]
        yWorld = geoInfoGrid[3] + row_cell * geoInfoGrid[5]

        [xWorld,yWorld] = np.meshgrid(xWorld,yWorld)
        coordCell = np.transpose(np.stack((np.ravel(xWorld),np.ravel(yWorld)),axis=0))

        return coordCell


    def getImageCoordByXYCoord(self, coord):
        """Return image coordinates for given coordinates.
        
        Parameters:
        -----------
        coord : np.array((n,2)) 
            List of coordinates to be evaluated

        Returns
        -------
        np.array()
            The coordinate of each cell of the map grid. 
            A N by 2 array with N is the number of cell, 1st col is x-coordinate, 
            2nd col is y-coordinate, The coordinate organized line by line.
        """
        geoInfoData = self.geoInfo

        imageCoord = np.zeros(coord.shape)

        imageCoord[:,0] = np.round((coord[:,0] - geoInfoData[0]) / geoInfoData[1])
        imageCoord[:,1] = np.round((geoInfoData[3] - coord[:,1]) / np.abs(geoInfoData[5]))

        return imageCoord.astype(int)     


    def getPatch(self, imageCoord, patchsize=32):
        """Return image patches for given coordinates and given patch size
        
        Parameters:
        -----------
        
        imageCoord: np.array((n,2))
            Array containing the image coordinated to get classified.

        patchSize: float
            Patch size of image patches (default is 32)

        Returns
        -------
        np.array((n, nBands, patchSize, ptachSize))
            Image patches for given coordinates.
        """
        # this function gets data patch with give image coordinate and patch size
        halfPatchSize = np.int(np.floor(patchsize/2))

        outData = np.lib.pad(self.data,((0,0),(halfPatchSize,halfPatchSize),(halfPatchSize,halfPatchSize)),'symmetric')
        outData = np.transpose(outData,(1,2,0))

        imageCoord = imageCoord + halfPatchSize

        print('INFO:    Array size: ' + str(imageCoord.shape[0]) + ',' + str(patchsize) + ',' + str(patchsize) + ',' + str(self.data.shape[0]))
        dataPatch = np.zeros((imageCoord.shape[0], patchsize,patchsize,self.data.shape[0]), dtype=np.float32)

        for i in range(0,imageCoord.shape[0]):
            dataPatch[i,:,:,:] = outData[imageCoord[i,1]-halfPatchSize:imageCoord[i,1]+halfPatchSize,imageCoord[i,0]-halfPatchSize:imageCoord[i,0]+halfPatchSize,:]

        return dataPatch

    def writeOutput(self, filename, data, type=np.int16, lczColor=False):
        bnd = data.shape[0]

        GDALDriver = gdal.GetDriverByName('GTiff')
        File = GDALDriver.Create(filename, self.col, self.row, bnd,
                     gdal_array.NumericTypeCodeToGDALTypeCode(type))
        File.SetProjection(self.proj)
        File.SetGeoTransform(self.geoInfo)

        # save file with int zeros
        idBnd = np.arange(0, bnd, dtype=int)
        for idxBnd in idBnd:
            outBand = File.GetRasterBand(int(idxBnd+1))
            if lczColor:
                outBand = applyLCZColors(outBand)
            outBand.WriteArray(data[idxBnd,:,:].astype(type))
            outBand.FlushCache()
            del(outBand)
        File = None


    def createEmptyFile(self, filename, nbBnd=17, xStride=100, yStride=-100, type=np.int16):
        """Create an empty GeoTiff initialized with zero values.
        
        Parameters:
        -----------
        
        filename : str
            Path to store file.

        nbBnd: float
            Numer of bands (default 17)
        
        xStride: float
            x stride in meters (default 100)

        yStride: float
            y stride in meters (default -100)

        type: np.dtype
            Data type to be used for storing (auto covnverted to GDAL type).
        """
        dataCoordSys = np.array(self.geoInfo)

        # set geoinformation for the output LCZ label grid
        # set resolution and coordinate for upper-left point
        LCZCoordSys = dataCoordSys.copy()
        LCZCoordSys[1] = xStride
        LCZCoordSys[5] = yStride
        LCZCol = np.arange(dataCoordSys[0],dataCoordSys[0]+self.col*dataCoordSys[1],LCZCoordSys[1]).shape[0]
        LCZRow = np.arange(dataCoordSys[3],dataCoordSys[3]+self.row*dataCoordSys[5],LCZCoordSys[5]).shape[0]

        # set the directory of initial grid & create directory if not existent
        savePath = '/'.join(filename.split('/')[:-1])
        if not os.path.exists(savePath):
            os.makedirs(savePath)

        LCZProb = np.zeros((nbBnd, LCZRow, LCZCol), dtype=type)

        writeGeoTiff(filename, LCZProb, self.proj, LCZCoordSys, type=type)


    def writeProbData(self, filename, data, nClasses=17):
        """Write class probabilities to file.
        
        Parameters:
        -----------
        
        filename : str
            Path to store file.

        data: np.array((N, nBands))
            Class probability for each pixel N. Is reshaped into image,
            scaled up by 10,000 and converted to int.
        
        nClasses: float
            number of LCZ classes

        """
        assert data.shape[0] == self.row * self.col
        # convert float to int and transpose
        data = np.array(data*1e4)
        data = data.astype(np.int16)
        data = np.transpose(np.reshape(data,(self.row, self.col, nClasses)), (2,0,1))
        self.writeOutput(filename, data)


def inferenceData(input_file, model_file, output_path=None, temperature=1.0, mixed=False, output_file_name=None):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"
    if mixed:
        mixed_precision.set_global_policy('mixed_float16')
    # 1. parameter setting
    # patch size
    patchsize = 32
    # patch shape
    patch_shape = (patchsize, patchsize, 10)
    # batch size
    batch_size = 256
    # max number of patches to be generated at once
    splitThresh=1e5
    if temperature != 1.0:
        activation=None
    else:
        activation='softmax'

    # 3. loading trained model
    model = resnet_v2.resnet_v2(input_shape=patch_shape, depth=20, num_classes=17, final_activation=activation)
    model.load_weights(path2NetModel)

    # initial classification map tiff file
    orgData = GDALHelper(city, readData=True, bands=[2,3,4,5,6,7,8,9,12,13], scale=10000.0)
    if output_file_name is None:
        cityname = city.rpartition('/')[-1].rpartition('_')[0]
    else:
        cityname = output_file_name
    outProbTif = outputPath + cityname + '_pro.tiff'
    orgData.createEmptyFile(outProbTif, type=np.int16)
    outLabelTif_mv = outputPath + cityname + '_lab.tiff'
    orgData.createEmptyFile(outLabelTif_mv, type=np.byte)
    probPredFile = GDALHelper(outProbTif)

    # get patch coordinate
    coordCell = probPredFile.getCoordLCZGrid()
    coordImage = orgData.getImageCoordByXYCoord(coordCell)

    # cutting patches
    nSplit = np.ceil(coordImage.shape[0] / splitThresh)
    probPred = []
    for split in range(0,int(nSplit)):
        coordImageBatch = coordImage[int(split*splitThresh):int((split+1)*splitThresh),:]
        dataPatches = orgData.getPatch(coordImageBatch, patchsize)
        # predict label
        pred = np.zeros((dataPatches.shape[0], 17))
        if(len(dataPatches[[not (entry==0).all() for entry in dataPatches]]) > 0):
            pred_tmp = model.predict(dataPatches[[not (entry==0).all() for entry in dataPatches]], batch_size=batch_size)
            if(activation is None):
                # Derive temperature scaled logits
                pred_tmp = tf.math.divide(pred_tmp, temperature)
                # Softmax transformation of scaled logits
                pred_tmp = tf.nn.softmax(pred_tmp).numpy()
            pred[[not (entry==0).all() for entry in dataPatches]] = pred_tmp

        del dataPatches
        probPred.append(pred)

    probPred = np.concatenate(probPred, axis=0)

    # 4. save predicted probability and label
    probPredFile.writeProbData(outProbTif, probPred)

    labelPredFile = GDALHelper(outLabelTif_mv)
    labelProb = np.reshape(probPred,(labelPredFile.row, labelPredFile.col, 17))
    labelPred = labelProb.argmax(axis=2).astype(np.uint8) + 1
    # set no_data value to LCZ class 0
    labelPred[(labelProb==0).all(axis=2)] = 0
    labelPredFile.writeOutput(outLabelTif_mv, np.expand_dims(labelPred, axis=0), lczColor=True, type=np.byte)

