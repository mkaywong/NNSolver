#!/usr/bin/env python
# coding: utf-8
import numpy as np
from Layer import Layer
from PassThrough import PassThrough
from Cost import Cost

class MeanRSS(Cost):
    '''
    Calculate the mean residual sum of squares.
    '''
    
    def createAndInitializeStruct(self):
        Cost.createAndInitializeStruct(self)
        self.z = np.zeros(self.inpShape, dtype = float)
        self.zi = np.zeros(self.inpShape[1:], dtype = float) 
        self.tmp = np.zeros(self.out.shape,dtype = float)
    
    def forward(self):
        np.copyto(self.out,self.bottom.getOutput())
        self.printDebugForward()
            
    def loss(self):
        self.Y = self.source.getY()
        self.inp = self.bottom.getOutput()
        c = Layer.currentBatchSize
        np.subtract(self.Y[:c],self.inp[:c],out=self.tmp[:c])
        np.square(self.tmp[:c],out=self.tmp[:c])
        L = np.sum(self.tmp[:c])
        if Layer.regularization:
            l2 = self.net.getParametersL2Sum()  # the factor 2 is for the entire L, thus no need here
            L = L + Layer.lmda*l2 
        L = L /(2*Layer.currentBatchSize)
        self.lossOut.append(L)
        return L
    
    def predict(self):
        return self.out

    def calcAccuracy(self):
        self.Y = self.source.getY()
        self.inp = self.bottom.getOutput()
        c = Layer.currentBatchSize
        np.subtract(self.Y[:c],self.inp[:c],out=self.tmp[:c])
        np.square(self.tmp[:c],out=self.tmp[:c])
        L = np.sum(self.tmp[:c])
        return L

    def backward(self):
        self.inp = self.bottom.getOutput()
        c = Layer.currentBatchSize
        np.subtract(self.inp[:c],self.Y[:c],out=self.inpDeriv[:c])
        np.divide(self.inpDeriv[:c],c,out=self.inpDeriv[:c])
        
        if Layer.currentBatchSize < Layer.batchSize:
            for i in range(Layer.currentBatchSize,Layer.batchSize):
                self.inpDeriv[i] = self.zi   
        self.printDebugBackward()        

