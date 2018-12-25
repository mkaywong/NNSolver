#!/usr/bin/env python
# coding: utf-8
import numpy as np
from Layer import Layer

class PassThrough(Layer):
          
    def __init__(self,para):
        Layer.__init__(self,para)
        
    def createAndInitializeStruct(self):        
        if self.top == None: 
            print(self, ' has None object as top.')
        self.createAndInitializeTopStruct()
        self.createAndInitializeBotStruct()            
        i = 1
        for j in self.inpShape[1:-1]:
            i = i * j
        
        self.s = i             # number of positions per output channel per sample
        
    def createAndInitializeTopStruct(self): 
        self.inp = self.bottom.getOutput()
        self.inpShape = self.inp.shape 
        self.outChannel = self.inpShape[-1]
        if self.top != None:
            if self.top.isPadded():   # only 2D layers are padded and cannot go from 1D to 2D....
                (pHOut,pWOut) = self.top.getPadShape()
                (_,hIn,wIn,cIn) = self.inpShape
                # out_shape is padded
                outShape = (Layer.batchSize,hIn+2*pHOut,wIn+2*pWOut,cIn)
                self.pOut = np.zeros(outShape,dtype=float)      # padded output
                # A is actual output. slice of pA
                self.out = self.pOut[:,pHOut:-pHOut,pWOut:-pWOut,:]
            else:
                self.out = np.zeros(self.inpShape,dtype=float)      
                # A is actual output. slice of pA
                self.pOut = self.out
        else:
            self.out = np.zeros(self.inpShape,dtype=float)
            
    
    def createAndInitializeBotStruct(self): 
        self.inpShape = self.bottom.getOutput().shape
        self.outChannel = self.inpShape[-1]
        # need to pad the derivatives of the input for bottom layer during back prop
        if self.bottom.isPadded():
            (pHIn,pWIn) = self.bottom.getPadShape()
            pInpDerivShape = (Layer.batchSize, self.inpShape[1]+2*pHIn, self.inpShape[2]+2*pWIn, self.inpShape[3])
            self.pInpDeriv = np.zeros(pInpDerivShape,dtype=float)                  # padded dJ / d input
            self.inpDeriv = self.pInpDeriv[:,pHIn:-pHIn,pWIn:-pWIn,:]
        else:
            self.inpDeriv = np.zeros(self.inpShape,dtype=float)                  # padded dJ / d input
            self.pIinpDeriv = self.inpDeriv
                    