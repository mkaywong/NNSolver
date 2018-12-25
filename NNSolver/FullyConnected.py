#!/usr/bin/env python
# coding: utf-8

import numpy as np
from Layer import Layer
from Linear import Linear

class FullyConnected(Linear):
    
    def __init__(self, Para):
        Linear.__init__(self, Para)
        try:
            self.outChannel = self.hPara['outChannel']
            self.bias = self.hPara['bias']
        except:
            self.printError("FullyConnected parameters does not exit or not complete.")
    
    def createAndInitializeStruct(self):
        self.inpShape = self.bottom.getOutput().shape
        t = self.inpShape[1:]
        self.inpNum = np.prod(t)
        self.inp = self.bottom.getOutput().reshape((Layer.batchSize,self.inpNum))
        self.WShape = (self.outChannel,self.inpNum)
        self.initParameters()       
        self.out = np.zeros((Layer.batchSize,self.outChannel),dtype=float)
        
        if self.bottom.isPadded():
            (ph,pw) = self.bottom.getPadShape()
            t = self.bottom.getOutput().shape
            pt = (t[0],t[1]+2*ph,t[2]+2*pw,t[3])
            self.pInpDeriv = np.zeros(pt,dtype = float)
            self.inpDeriv = self.pInpDeriv[:,ph:-ph,pw:-pw,:].reshape((pt[0],-1))
        else:
            t = self.bottom.getOutput().shape
            self.pInpDeriv = np.zeros(t,dtype = float)
            self.inpDeriv = self.pInpDeriv.reshape((t[0],-1))

        self.createOptimizerStruct()
        
    def forward(self):
        self.inp = self.bottom.getOutput().reshape((Layer.batchSize,self.inpNum))
        np.dot(self.inp[:Layer.currentBatchSize,:],self.W.T,out=self.out[:Layer.currentBatchSize])
        # In matrix computation, b^T should be added to A, however, numpy broadcasting tread them the same
        # way, so, no transpose is needed.
        if self.bias:
            np.add(self.out[:Layer.currentBatchSize,:], self.b, out = self.out[:Layer.currentBatchSize,:])
        self.printDebugForward()
    
    def backward(self):
        self.inp = self.bottom.getOutput().reshape((Layer.batchSize,self.inpNum))
        dA = self.top.getInputDerivative()
        # compute dJ/d_inp
        np.dot(dA[0:Layer.currentBatchSize],self.W,out=self.inpDeriv[0:Layer.currentBatchSize])
        # compute dJ/dw
        np.dot((dA[0:Layer.currentBatchSize]).T,self.inp[0:Layer.currentBatchSize],out=self.dW)
        # compute dJ/db
        if self.bias:
            np.sum(dA[0:Layer.currentBatchSize],axis=0,out=self.db)
        
        if Layer.regularization:
            np.add(self.dW, (Layer.lmda/Layer.currentBatchSize)*self.W, out=self.dW)
            if self.bias: np.add(self.db, (Layer.lmda/Layer.currentBatchSize)*self.b, out=self.db)
        
        self.printDebugBackward()
    
    def getInputDerivative(self):
        return self.inpDeriv.reshape(self.inpShape)

