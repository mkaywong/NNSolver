#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as pt

def ImageToArray(batchVol,kernalShape,stride,outArray):
    """
      This procedure convert a 4D volume into a 2D array for convolution computation.
      Row major is assumed. That is, data contiquous along the channel C in memory.
      
      batchVol: A batch of 3 D volume - batch size N x height H x width W x channel C
                batchVol needs to be padded accordingly before calling this routine
      kernalShape: (k_h,k_w)
      stride: distance between kernal application, same distance for height and width dimension
      outArray: array to store the converted 2D array
    """
    N,H,W,C = batchVol.shape
    k_h,k_w = kernalShape
    H_out = int((H-k_h)/stride)+1
    W_out = int((W-k_w)/stride)+1
    ex_h = N*H_out*W_out
    ex_w = k_h*k_w*C
    if outArray.shape != (ex_h,ex_w):
        print("ImageToArray: outArray shape ",outArray.shape, 
              " does not match with expected dims (",ex_h,',',ex_w,')')
        assert False
    r = 0
    for i in range(N):
        for j in range(0,H-k_h+1,stride):
            for k in range(0,W-k_w+1,stride):
                for l in range(k_h):
                    st_col = (l*k_w)*C
                    end_col = st_col+k_w*C
                    outArray[r,st_col:end_col] = batchVol[i,j+l,k:k+k_w].reshape((1,k_w*C))
                r = r+1

class colors:
    """ Simple ANSI escape sequences """
    BOLD = '\033[1m'
    RED = '\033[91m'
    END = '\033[0m'

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

def plotCifar10Images(image,labelOneH,labelNames,idx):
    fig,ax = pt.subplots(2,8,figsize=(8,2))
    fig.tight_layout()
    for i in range(16):
        ax[i//8,i%8].imshow(image[idx+i])
        ax[i//8,i%8].set_title(labelNames[np.argmax(labelOneH[idx+i])])
        ax[i//8,i%8].axis('off')