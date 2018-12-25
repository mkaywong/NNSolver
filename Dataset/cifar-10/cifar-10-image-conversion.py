#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as pt

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def convertCifar10ImShape(image):
    result = np.zeros((image.shape[0],32,32,3),dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(32):
            for k in range(32):
                for l in range(3):
                    result[i,j,k,l] = image[i,l*1024+j*32+k]
    return result

def convertCifar10Label2OneHot(labels):
    labelOneHot = np.zeros((labels.shape[0],10),dtype=np.uint8)
    idx = np.arange(labels.shape[0])
    labelOneHot[idx,labels] = 1
    return labelOneHot

def plotCifar10Images(image,labelOneH,labelNames,idx):
    fig,ax = pt.subplots(2,8,figsize=(8,2))
    fig.tight_layout()
    for i in range(16):
        ax[i//8,i%8].imshow(image[idx+i])
        ax[i//8,i%8].set_title(labelNames[np.argmax(labelOneH[idx+i])])
        ax[i//8,i%8].axis('off')

# The original data files are downloaded from https://www.cs.toronto.edu/~kriz/cifar.html
# Update the file locations before running the file.

f1 = '../Dataset/cifar-10-batches-py/data_batch_1'
f2 = '../Dataset/cifar-10-batches-py/data_batch_2'
f3 = '../Dataset/cifar-10-batches-py/data_batch_3'
f4 = '../Dataset/cifar-10-batches-py/data_batch_4'
f5 = '../Dataset/cifar-10-batches-py/data_batch_5'
ftest = '../Dataset/cifar-10-batches-py/test_batch'
f_meta = '../Dataset/cifar-10-batches-py/batches.meta'

im1 = unpickle(f1)
im2 = unpickle(f2)
im3 = unpickle(f3)
im4 = unpickle(f4)
im5 = unpickle(f5)
imtest = unpickle(ftest)
meta = unpickle(f_meta)

print('data batch: ', im1.keys())
print('meta: ', meta.keys())

labelNames = []
for i in range(len(meta[b'label_names'])):
    labelNames.append(meta[b'label_names'][i].decode("utf-8"))
print(labelNames)

type(im1[b'data'][0,0])

cifar10TrainImages = convertCifar10ImShape(np.stack([im1[b'data'],im2[b'data'],im3[b'data'],im4[b'data'],
                                                     im5[b'data']]).reshape(50000,-1))
cifar10TrainLabels = convertCifar10Label2OneHot(np.stack([np.array(im1[b'labels']),np.array(im2[b'labels']),
                            np.array(im3[b'labels']),np.array(im4[b'labels']),np.array(im5[b'labels'])]).reshape(50000))
cifar10TestImages = convertCifar10ImShape(imtest[b'data'])
cifar10TestLabels = convertCifar10Label2OneHot(np.array(imtest[b'labels']))

print(cifar10TrainImages.shape, cifar10TrainLabels.shape, cifar10TestImages.shape, cifar10TestLabels.shape)

idx = 5000
plotCifar10Images(cifar10TestImages,cifar10TestLabels,labelNames,idx)

cifar10Data = {}
cifar10Data['trainImages'] = cifar10TrainImages
cifar10Data['trainLabels'] = cifar10TrainLabels
cifar10Data['testImages'] = cifar10TestImages
cifar10Data['testLabels'] = cifar10TestLabels
cifar10Data['labelNames'] = labelNames
np.savez('cifar10Data.npz',**cifar10Data)

data = np.load('cifar10Data.npz')

idx = 5000
plotCifar10Images(data['testImages'],data['testLabels'],data['labelNames'],idx)

