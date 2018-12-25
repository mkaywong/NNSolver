#!/usr/bin/env python
# coding: utf-8

import numpy as np
from Layer import Layer

class ImageDataSource(Layer):
    
    srcDimList = [(0),(0,1),(0,1,2)]
    
    def __init__(self,para):
        Layer.__init__(self,para)
        self.lastMiniBatch = True
                
    def createAndInitializeStruct(self):
        if self.top.isPadded():  
            (pHOut,pWOut) = self.top.getPadShape()
            (hIn,wIn,cIn) = self.imageDim
            # out_shape is padded
            outShape = (Layer.batchSize,hIn+2*pHOut,wIn+2*pWOut,cIn)
            self.pOut = np.zeros(outShape,dtype=float)      # padded output
            # out is actual output. slice of pOut
            self.out = self.pOut[:,pHOut:-pHOut,pWOut:-pWOut,:]
        else:
            self.out = np.zeros((Layer.batchSize,)+self.imageDim,dtype=float)      
            # out is actual output. slice of pOut
            self.pOut = self.out
        self.label = np.zeros((Layer.batchSize,)+self.labelDim,dtype=float)
        self.zeros = np.zeros(self.imageDim)
        self.zerosL = np.zeros(self.labelDim)
    
    def setTrainData(self,image,label):
        self.trainImage = image
        self.trainLabel = label
        self.trainNum = self.trainImage.shape[0]
        self.imageDim = self.trainImage.shape[1:]
        self.labelDim = self.trainLabel.shape[1:]
        self.sumAxis = ImageDataSource.srcDimList[len(self.trainImage.shape)-2]
        self.trainImageMean = np.mean(self.trainImage,axis=self.sumAxis)
        self.trainImageStd = np.std(self.trainImage,axis=self.sumAxis)        
        self.trainImageNorm = (self.trainImage - self.trainImageMean)/self.trainImageStd
    
    def setTestData(self,image,label):
        self.testImage = image
        self.testLabel = label
        self.testNum = self.testImage.shape[0]
        self.testImageNorm = (self.testImage - self.trainImageMean)/self.trainImageStd
    
    def useTrainData(self):
        self.currentImageNum = self.trainNum
        self.currentImage = self.trainImageNorm
        self.currentLabel = self.trainLabel
        self.lastMiniBatch = True
    
    def useTestData(self):
        self.currentImageNum = self.testNum
        self.currentImage = self.testImageNorm
        self.currentLabel = self.testLabel
        self.lastMiniBatch = True      
    
    def getY(self):
        return self.label
    
    def forward(self):
        Layer.currentBatchSize = Layer.batchSize
        if self.lastMiniBatch:
            self.lastMiniBatch = False
            if not Layer.inferenceMode:
                self.perm = np.random.permutation(self.currentImageNum)
            else:
                self.perm = np.arange(self.currentImageNum)
            self.currentStart = 0
            self.sampleCount = 0
        if (self.currentStart + Layer.batchSize) <= self.currentImageNum:
            j = 0
            for i in self.perm[self.currentStart:self.currentStart+Layer.batchSize]:
                self.out[j] = self.currentImage[i]
                self.label[j] = self.currentLabel[i]
                j = j+1
            self.currentStart = self.currentStart + Layer.batchSize
            if self.currentStart == self.currentImageNum:
                self.lastMiniBatch = True 
        else:
            j = 0
            for i in self.perm[self.currentStart:]:
                self.out[j] = self.currentImage[i]
                self.label[j] = self.currentLabel[i]
                j = j+1
            Layer.currentBatchSize = j
            for k in range(j,Layer.batchSize):
                self.out[k] = self.zeros
                self.label[k] = self.zerosL
            self.lastMiniBatch = True
        self.sampleCount = self.sampleCount + Layer.currentBatchSize

    def currentDataSetSize(self):
        return self.currentImageNum

    def endOfEpoch(self):
        return self.lastMiniBatch
