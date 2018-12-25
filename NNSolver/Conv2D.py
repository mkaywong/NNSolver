#!/usr/bin/env python
# coding: utf-8
import numpy as np
import math
from Utilities import ImageToArray
from Layer import Layer
from Linear import Linear

class Conv2D(Linear):
        
    def __init__(self, Para):
        Linear.__init__(self,Para)
        try:
            self.padding = Para['padding']
            if self.padding:
                self.padShape = Para['padShape']   # 2D tuple (pH,pW) padding on each side
            else:
                self.padShape = (0,0)
            self.outChannel = Para['outChannel']
            self.kernelShape = Para['kernelShape'] # should be a 2D tuple (kH,kW)
            self.stride = Para['stride'] # same for height and width dimension
            self.bias = Para['bias']
        except:
            self.printError("Conv2D parameters does not exit or not complete.")
    
    def createAndInitializeStruct(self):
        self.inpShape = self.bottom.getOutput().shape
        self.inpNum = np.prod(self.inpShape[1:])
        (_,hIn,wIn,cIn) = self.inpShape
        (pH,pW) = self.padShape
        (kH,kW) = self.kernelShape
        hOut = (hIn + 2*pH - kH)//self.stride + 1
        wOut = (wIn + 2*pW - kW)//self.stride + 1
        
        if self.top.isPadded():
            (pHOut,pWOut) = self.top.getPadShape()
            # out_shape is padded
            outShape = (Layer.batchSize,hOut+2*pHOut,wOut+2*pWOut,self.outChannel)
            self.pOut = np.zeros(outShape,dtype=float)      # padded output
            # out is actual output. slice of pOut
            self.out = self.pOut[:,pHOut:-pHOut,pWOut:-pWOut,:]
        else:
            outShape = (Layer.batchSize,hOut,wOut,self.outChannel)
            self.pOut = np.zeros(outShape,dtype=float)      
            # out is actual output. slice of pOut
            self.out = self.pOut
        
        # 2 D version of output, to store the results of convolution
        self.A2D = self.out.reshape((Layer.batchSize*hOut*wOut,self.outChannel))
        # array for image-to-array operation
        self.im2arr = np.zeros((Layer.batchSize*hOut*wOut,kH*kW*cIn),dtype=float)
        self.outputArea = hOut*wOut        
        self.WShape = (self.outChannel,self.kernelShape[0],self.kernelShape[1],cIn)
        
        self.initParameters()
                
        # WBackprop stores the weight matrix for backprop computation
        self.WBackprop = np.zeros((cIn,self.kernelShape[0],self.kernelShape[1],self.outChannel),dtype=float)
        
        # need to pad the derivatives of the input for bottom layer during back prop
        if self.bottom.isPadded():
            (pHIn,pWIn) = self.bottom.getPadShape()
            pInpDerivShape = (Layer.batchSize, self.inpShape[1]+2*pHIn, self.inpShape[2]+2*pWIn, self.inpShape[3])
            self.pInpDeriv = np.zeros(pInpDerivShape,dtype=float)                  # padded dJ / d input
            self.inpDeriv = self.pInpDeriv[:,pHIn:-pHIn,pWIn:-pWIn,:]
        else:
            self.pInpDeriv = np.zeros(self.inpShape,dtype=float)                  # padded dJ / d input
            self.inpDeriv = self.pInpDeriv
            
        # dAExpanded is used to stored the expanded output derivatives before backprop conv
        if self.stride > 1:
            dAEShape = (Layer.batchSize, self.inpShape[1]+2*pH,self.inpShape[2]+2*pW,self.outChannel)
            self.dAExpanded = np.zeros(dAEShape,dtype=float)
                
        # 2 D version of input derivative, to store the results of convolution for backprop
        self.inpDeriv2d = self.inpDeriv.reshape((Layer.batchSize*hIn*wIn,cIn))
        # array for image-to-array operation
        self.im2arrBackprop = np.zeros((Layer.batchSize*hIn*wIn,kH*kW*self.outChannel),dtype=float)
        self.inputArea =hIn*wIn
        
        self.createOptimizerStruct()
        
    def forward(self):
        if self.padding:
            self.inp = self.bottom.getPaddedOutput()
        else:
            self.inp = self.bottom.getOutput()
        i2a = self.im2arr[:Layer.currentBatchSize*self.outputArea,:]
        ImageToArray(self.inp[:Layer.currentBatchSize,:,:,:],self.kernelShape,self.stride,i2a)
        np.dot(i2a,self.W.reshape((self.outChannel,-1)).T,out=self.A2D[:Layer.currentBatchSize*self.outputArea,:])
        if self.bias: np.add(self.out[:Layer.currentBatchSize],self.b,out=self.out[:Layer.currentBatchSize])    
        self.printDebugForward()
    
    def backward(self):
        dA = self.top.getInputDerivative()
        (N,h,w,c) = dA.shape
        dAR = dA.reshape((N*h*w,c))
        t = Layer.currentBatchSize*h*w
        # reuse the im2arr built in the forward path. 
        np.dot(dAR[:t].T,self.im2arr[:t], 
               out=self.dW.reshape(self.outChannel,-1))
        # calculate db
        if self.bias: 
            np.sum(dA[:Layer.currentBatchSize],axis=(0,1,2),out=self.db)
        if Layer.regularization:
            np.add(self.dW, (Layer.lmda/Layer.currentBatchSize)*self.W, out=self.dW)
            if self.bias:
                np.add(self.db, (Layer.lmda/Layer.currentBatchSize)*self.b, out=self.db)
            
        # calculate d_inp backward convolution
        # W matrix in backward prop
        (cout,h,w,cin) = self.W.shape
        for i in range(cin):
            for j in range(cout):
                self.WBackprop[i,:,:,j] = np.rot90(self.W[j,:,:,i],2)

        if self.padding: 
            dA = self.top.getPInputDerivative()

        if self.stride > 1:   # need to expand the array by filling in rows and cols of zeros
            gap = self.stride -1
            tmpDA = self.top.getInputDerivative()
            (N,h,w,c) = tmpDA.shape
            for i in range(Layer.currentBatchSize):
                for j in range(h):
                    for k in range(w):
                        self.dAExpanded[i,self.padShape[0]+j+j*gap,self.padShape[1]+k+k*gap,:] = tmpDA[i,j,k,:]
            dA = self.dAExpanded
            
        i2a = self.im2arrBackprop[:Layer.currentBatchSize*self.inputArea,:]
        ImageToArray(dA[:Layer.currentBatchSize,:,:,:],self.kernelShape,1,i2a)
        np.dot(i2a,self.WBackprop.reshape((cin,-1)).T,out=self.inpDeriv2d[:Layer.currentBatchSize*self.inputArea,:])    
        self.printDebugBackward()

    def printAllShape(self):
        Linear.printIOShape(self)
        if self.stride > 1: print('dAExpanded\n',self.dAExpanded.shape)

