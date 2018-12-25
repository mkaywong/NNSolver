#!/usr/bin/env python
# coding: utf-8
import numpy as np
from Layer import Layer
from PassThrough import PassThrough

class Normalize(PassThrough):
    '''
    Normalize output the batch normalized input. It keeps a history (numStats) of pass mean and std values.
    During inference mode, the mean of pass norms are used instead of batch norm.
    '''
    normDimList=[(0),(0,1),(0,1,2),(0,1,2,3)]
    epsilon = 0.0001
    numStats = 100
        
    def __init__(self,para):
        PassThrough.__init__(self,para)
            
    def createAndInitializeStruct(self):
        PassThrough.createAndInitializeStruct(self)
        self.normD = Normalize.normDimList[len(self.inpShape)-2]     # axises to calculate mean and std
        self.mean = np.zeros(self.outChannel,dtype=float)
        self.var = np.zeros(self.outChannel,dtype=float)
        self.std = np.zeros(self.outChannel,dtype=float)
        self.dMean = np.zeros(self.outChannel,dtype=float)
        self.dVar = np.zeros(self.outChannel,dtype=float)
        self.meanList=np.zeros((Normalize.numStats,self.outChannel),dtype=float)
        self.stdList=np.zeros((Normalize.numStats,self.outChannel),dtype=float)
        self.tmp = np.zeros(self.inpShape,dtype=float)
        self.tmp2 = np.zeros(self.outChannel,dtype=float)
        self.count = 0
        self.normComputed = False
    
    def setMeanStd(self,m,sd):
        if self.mean.shape == m.shape:
            np.copyto(self.mean,m)
            if self.std.shape == sd.shape:
                np.copyto(self.std,sd)
            else:
                print('sd:',sd.shape,' self.std:',self.std.shape)
                self.printError('shape mismatch while setting std')
        else:
            print('m:',m.shape,'self.mean:',self.mean.shape)
            self.printError('shape mismatch while setting mean')
    
    def forward(self):
        inp = self.bottom.getOutput()
        c = Layer.currentBatchSize
        if not Layer.inferenceMode:
            np.mean(inp[:c],axis=self.normD,out=self.mean)
            np.var(inp[:c],axis=self.normD,out=self.var)
            np.add(self.var,Normalize.epsilon,out=self.var)
            np.sqrt(self.var,out=self.std)
            self.meanList[self.count] = self.mean
            self.stdList[self.count] = self.std
            self.count = (self.count + 1)%Normalize.numStats
            self.normComputed = False
        elif not self.normComputed:
            np.mean(self.meanList,axis=0,out=self.mean)
            np.mean(self.stdList,axis=0,out=self.std)
            self.normComputed = True
        np.subtract(inp[:c],self.mean,out=self.out[:c])
        np.divide(self.out[:c],self.std,out=self.out[:c])
            
        self.printDebugForward()
    
    def saveParameters(self, pD):
        np.mean(self.meanList,axis=0,out=self.mean)
        np.mean(self.stdList,axis=0,out=self.std)
        pD[self.instanceName+':mean'] = self.mean
        pD[self.instanceName+':std'] = self.std

    def loadParameters(self, pD):
        try:
            np.copyto(self.mean,pD[self.instanceName+':mean'])
            np.copyto(self.var,pD[self.instanceName+':std'])
        except:
            self.printError("Layer parameter(s) does not exist while loading.")

    def backward(self):
        inp = self.bottom.getOutput()
        dA = self.top.getInputDerivative()
        c = Layer.currentBatchSize
        self.m = self.s*c
        # dl/dvar
        np.multiply(dA[:c],self.out[:c],out=self.inpDeriv[:c])
        np.sum(self.inpDeriv[:c],axis=self.normD,out=self.dVar)
        np.divide(self.dVar,-2*self.var,out=self.dVar)
        
        # dl/dmean
        np.sum(dA[:c],axis=self.normD,out=self.dMean)
        np.divide(self.dMean,-self.std,out=self.dMean)
        
        # dl/dx
        np.divide(dA[:c],self.std,out=self.inpDeriv[:c])

        np.subtract(inp[:c],self.mean,out=self.tmp[:c])
        np.multiply(self.dVar,2/self.m,out=self.tmp2)
        np.multiply(self.tmp[:c],self.tmp2,out=self.tmp[:c])
        
        np.multiply(self.dMean,1/self.m,out=self.tmp2)
        
        np.add(self.inpDeriv[:c],self.tmp[:c],out=self.tmp[:c])        
        np.add(self.tmp[:c],self.tmp2,out=self.inpDeriv[:c])
        
        self.printDebugBackward()
