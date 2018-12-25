#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as pt

dirName = '../Dataset/MNIST/'
trainImageFile = "train-images.idx3-ubyte"
trainLabelFile = "train-labels.idx1-ubyte"
testImageFile = "t10k-images.idx3-ubyte"
testLabelFile = "t10k-labels.idx1-ubyte"

def readMNISTImage(fileName):
    infile = open(fileName,'rb')
    magic = int.from_bytes(infile.read(4),byteorder='big')
    assert magic == 2051, 'File magic number not equal to 2051'
    totalImageNum = int.from_bytes(infile.read(4),byteorder='big')
    W = int.from_bytes(infile.read(4),byteorder='big')
    H = int.from_bytes(infile.read(4),byteorder='big')
    image = np.zeros((totalImageNum,H,W,1),dtype=int)
    imageSize = W * H
    for i in range(totalImageNum):
        image[i,:,:,:] = np.frombuffer(infile.read(imageSize),dtype=np.uint8).reshape((1,H,W,1))
    infile.close()
    return image

def readMNISTLabel(fileName):
    infile = open(fileName,'rb')
    magic = int.from_bytes(infile.read(4),byteorder='big')
    assert magic == 2049, 'File magic number not equal to 2049'
    totalLabelNum = int.from_bytes(infile.read(4),byteorder='big')
    label = np.frombuffer(infile.read(totalLabelNum),dtype=np.uint8)
    infile.close()
    labelOneHot = np.zeros((totalLabelNum,10),dtype=float)
    idx = np.arange(totalLabelNum)
    labelOneHot[idx,label] = 1
    return labelOneHot

def plotMNISTResult(data,label,idx):
    '''
    data: numpy array containing MNIST images (,28,28,1)
    label: numpy array containing one hot vectors (,10)
    idx: an array of indexes to plot, the first 16 indexes are used
    '''
    print(np.argmax(label[idx],axis=1))
    _, ax = pt.subplots(1,16,figsize=(10,1))
    for i in range(16):
        ax[i].axis('off')
        ax[i].imshow(data[idx[i]].reshape((28,28)))

# The original data is downloaded from http://yann.lecun.com/exdb/mnist/
# Update the location and the names of the files.

mnistTrainData = readMNISTImage(dirName+trainImageFile)
mnistTrainLabel = readMNISTLabel(dirName+trainLabelFile)
mnistTestData = readMNISTImage(dirName+testImageFile)
mnistTestLabel = readMNISTLabel(dirName+testLabelFile)

mnistData = {}
mnistData['trainImages'] = mnistTrainData
mnistData['trainLabels'] = mnistTrainLabel
mnistData['testImages'] = mnistTestData
mnistData['testLabels'] = mnistTestLabel
np.savez('mnistData.npz',**mnistData)

data = np.load('mnistData.npz')

print(data['testImages'].shape, data['testLabels'].shape)

plotMNISTResult(data['testImages'],data['testLabels'],np.arange(32,48))
