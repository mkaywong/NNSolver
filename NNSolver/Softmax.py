#!/usr/bin/env python
# coding: utf-8

import numpy as np
from Layer import Layer
from Cost import Cost

class Softmax(Cost):
    """
    Softmax 
    """
    def createAndInitializeStruct(self):
        Cost.createAndInitializeStruct(self)  
        self.z = np.zeros(self.inpShape[1:])
        self.sumD = Cost.axisDim[len(self.inpShape)-2]
    
    def forward(self):
        self.inp = self.bottom.getOutput()
        np.exp(self.inp[:Layer.currentBatchSize],out=self.out[:Layer.currentBatchSize])
        tmp = np.sum(self.out[0:Layer.currentBatchSize],axis=self.sumD).reshape((Layer.currentBatchSize,1))
        np.divide(self.out[0:Layer.currentBatchSize].reshape((Layer.currentBatchSize,-1)),tmp,
                 out=self.out[0:Layer.currentBatchSize].reshape((Layer.currentBatchSize,-1)))
        if Layer.currentBatchSize < Layer.batchSize:
            for i in range(Layer.currentBatchSize,Layer.batchSize):
                self.out[i] = self.z
        self.printDebugForward()
            
    def loss(self):
        """
        Return the mini-batch loss.
        """
        self.Y = self.source.getY()
        L = - np.sum((self.Y[0:Layer.currentBatchSize]*np.log(self.out[0:Layer.currentBatchSize])))
        # for regularization, need to include the components of the parameters
        if Layer.regularization:
            l2 = self.net.getParametersL2Sum()/2
            L = L + Layer.lmda*l2 
        L = L /Layer.currentBatchSize
        self.lossOut.append(L)
        return L

    def predict(self):
        '''
        Return the class number for the mini-batch
        '''
        return np.argmax(self.out[:Layer.currentBatchSize],axis=1)

    def calcAccuracy(self):
        """
        Return the number of correct classification within the mini-batch
        """
        self.Y = self.source.getY()
        pred = np.argmax(self.out[:Layer.currentBatchSize],axis=1)
        return np.sum(self.Y[np.arange(Layer.currentBatchSize),pred])

    def backward(self):
        c = Layer.currentBatchSize
        np.subtract(self.out[0:c],self.Y[0:c],out=self.inpDeriv[0:c])
        np.divide(self.inpDeriv[0:c], c, out=self.inpDeriv[0:c])
        if Layer.currentBatchSize < Layer.batchSize:
            for i in range(Layer.currentBatchSize,Layer.batchSize):
                self.inpDeriv[i] = self.z   
        self.printDebugBackward()
