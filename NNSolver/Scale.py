#!/usr/bin/env python
# coding: utf-8

import numpy as np
from Layer import Layer
from PassThrough import PassThrough

class Scale(PassThrough):
    
    normDimList=[(0),(0,1),(0,1,2),(0,1,2,3)]
    
    def __init__(self,para):
        PassThrough.__init__(self,para)
        
    def createAndInitializeStruct(self):
        PassThrough.createAndInitializeStruct(self)
        # set bias to zeros
        self.sBeta = np.zeros(self.outChannel,dtype=float)       # bias intialized to zeros
        self.sGamma = np.ones(self.outChannel,dtype=float)       # Gamma initialized to ones
        self.normD = Scale.normDimList[len(self.inpShape)-2]     # axises to calculate mean and std
        self.dGamma = np.zeros(self.outChannel,dtype=float)
        self.dBeta = np.zeros(self.outChannel,dtype=float)
        self.paraDict['sBeta'] = self.sBeta
        self.paraDict['sGamma'] = self.sGamma
        self.dParaDict['sBeta'] = self.dBeta
        self.dParaDict['sGamma'] = self.dGamma
        self.createOptimizerStruct()
        
    def forward(self):
        inp = self.bottom.getOutput()
        c = Layer.currentBatchSize
        np.multiply(inp[:c],self.sGamma,out=self.out[:c])
        np.add(self.out[:c],self.sBeta,out=self.out[:c])
        self.printDebugForward()
        
    def backward(self):        
        dA = self.top.getInputDerivative()
        inp = self.bottom.getOutput()
        c = Layer.currentBatchSize
        np.multiply(dA[:c],inp[:c],out=self.inpDeriv[:c])
        np.sum(self.inpDeriv[:c],axis=self.normD,out=self.dGamma)
        np.sum(dA[:c],axis=self.normD,out=self.dBeta)
        np.multiply(dA[:c],self.sGamma,out=self.inpDeriv[:c])
        self.printDebugBackward()

