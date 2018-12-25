#!/usr/bin/env python
# coding: utf-8
import numpy as np
from Layer import Layer
from PassThrough import PassThrough

class Pool(PassThrough):
    
    poolTypes = ['max','ave']
    
    def __init__(self, para):
        PassThrough.__init__(self, para)
        try:
            self.poolType = para['poolType']
            if not (self.poolType in self.poolTypes):
                self.printError('Pool type not defined.')
            self.stride = para['stride']
            self.kernelShape = para['kernelShape']
        except:
            self.printError('Parameters not complete.')
    
    def createAndInitializeStruct(self):
        self.inpShape = self.bottom.getOutput().shape
        (_,hIn,wIn,cIn) = self.inpShape
        (kH,kW) = self.kernelShape
        self.kArea = kH*kW
        hOut = (hIn - kH)//self.stride + 1
        wOut = (wIn - kW)//self.stride + 1 
        self.outChannel = cIn
        self.outShape = (Layer.batchSize,hOut,wOut,self.outChannel)
        self.zerosArray = np.zeros(self.inpShape[1:],dtype=float)
        
        if self.top.isPadded():
            (pHOut,pWOut) = self.top.getPadShape()
            pOutShape = (Layer.batchSize,self.outShape[1]+2*pHOut,self.outShape[2]+2*pWOut,self.outChannel)
            self.pOut = np.zeros(pOutShape,dtype=float)      # padded activations
            self.out = self.pOut[:,pHOut:-pHOut,pWOut:-pWOut,:]
        else:
            self.pOut = np.zeros(self.outShape,dtype=float)      
            self.out = self.pOut
            
        self.tmpDA = np.zeros(self.outShape,dtype=float)        
        PassThrough.createAndInitializeBotStruct(self)
    
    def forward(self):
        if self.poolType == 'max':
            self.maxForward()
        if self.poolType == 'ave':
            self.aveForward()
        self.printDebugForward()
    
    def maxForward(self):
        inp = self.bottom.getOutput()[:Layer.currentBatchSize]
        for i in range(Layer.currentBatchSize):
            jj = 0
            for j in range(self.kernelShape[0],self.inpShape[1]+1,self.stride):
                jSt = j - self.kernelShape[0]
                kk = 0
                for k in range(self.kernelShape[1],self.inpShape[2]+1,self.stride):
                    kSt = k - self.kernelShape[1]
                    self.out[i,jj,kk,:] = np.max(inp[i,jSt:j,kSt:k,:],axis=(0,1))
                    kk = kk + 1                        
                jj = jj + 1
                                
    def aveForward(self):
        inp = self.bottom.getOutput()[:Layer.currentBatchSize]
        for i in range(Layer.currentBatchSize):
            jj = 0
            for j in range(self.kernelShape[0],self.inpShape[1]+1,self.stride):
                jSt = j - self.kernelShape[0]
                kk = 0
                for k in range(self.kernelShape[1],self.inpShape[2]+1,self.stride):
                    kSt = k - self.kernelShape[1]
                    self.out[i,jj,kk,:] = np.mean(inp[i,jSt:j,kSt:k,:],axis=(0,1))
                    kk = kk + 1                        
                jj = jj + 1
       
    def backward(self):
        for i in range(Layer.currentBatchSize):
            np.copyto(self.inpDeriv[i],self.zerosArray)
        if self.poolType == 'max':
            self.maxBackward()
        if self.poolType == 'ave':
            self.aveBackward()
        self.printDebugBackward()
    
    def maxBackward(self):
        inp = self.bottom.getOutput()[:Layer.currentBatchSize]
        dA = self.top.getInputDerivative()[:Layer.currentBatchSize]
        for i in range(Layer.currentBatchSize):
            jj = 0
            for j in range(self.kernelShape[0],self.inpShape[1]+1,self.stride):
                jSt = j - self.kernelShape[0]
                kk = 0
                for k in range(self.kernelShape[1],self.inpShape[2]+1,self.stride):
                    kSt = k - self.kernelShape[1]
                    for c in range(self.inpShape[3]):
                        am = np.argmax(inp[i,jSt:j,kSt:k,c])
                        self.inpDeriv[i,jSt+am//self.kernelShape[1],kSt+am%self.kernelShape[1],c] =                         self.inpDeriv[i,jSt+am//self.kernelShape[1],kSt+am%self.kernelShape[1],c] + dA[i,jj,kk,c]
                    kk = kk + 1                        
                jj = jj + 1
                        
    def aveBackward(self):
        np.divide(self.top.getInputDerivative()[:Layer.currentBatchSize],self.kArea,
                  out=self.tmpDA[:Layer.currentBatchSize])
        for i in range(Layer.currentBatchSize):
            jj = 0
            for j in range(self.kernelShape[0],self.inpShape[1]+1,self.stride):
                jSt = j - self.kernelShape[0]
                kk = 0
                for k in range(self.kernelShape[1],self.inpShape[2]+1,self.stride):
                    kSt = k - self.kernelShape[1]
                    np.add(self.inpDeriv[i,jSt:j,kSt:k,:],self.tmpDA[i,jj,kk,:],
                           out=self.inpDeriv[i,jSt:j,kSt:k,:])
                    kk = kk + 1                        
                jj = jj + 1
       
