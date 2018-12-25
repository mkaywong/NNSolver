#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math
from Layer import Layer
from PassThrough import PassThrough

__all__ = ['Activation']

class Activation(PassThrough):
    
    activationTypes = ['ReLU','Sigmoid']
    
    def __init__(self, Para):
        PassThrough.__init__(self,Para)
        try:
            self.activationType = Para['activationType']   
            if not (self.activationType in Activation.activationTypes):
                self.printError('Activation type not defined')
        except:
            self.printError('Activation type not defined')
    
    def forward(self):
        inp = self.bottom.getOutput()
        if self.activationType == 'ReLU':
            self.ReLUForward(inp[:Layer.currentBatchSize],self.out[:Layer.currentBatchSize])
        elif self.activationType == 'Sigmoid':
            self.SigmoidForward(inp[:Layer.currentBatchSize],self.out[:Layer.currentBatchSize])
        self.printDebugForward()
    
    def backward(self):
        inp = self.bottom.getOutput()
        dA = self.top.getInputDerivative()
        if self.activationType == 'ReLU':
            self.ReLUBackward(inp[:Layer.currentBatchSize],self.inpDeriv[:Layer.currentBatchSize])
        elif self.activationType == 'Sigmoid':
            self.SigmoidBackward(self.out[:Layer.currentBatchSize],self.inpDeriv[:Layer.currentBatchSize])
        np.multiply(self.inpDeriv[:Layer.currentBatchSize], dA[:Layer.currentBatchSize], 
                    out=self.inpDeriv[:Layer.currentBatchSize])
        self.printDebugBackward()

    #############################        
    # for ReLU    
    def ReLUForward(self,a,b):
        np.maximum(a,0.,out=b)
        
    def ReLUBackward(self,a,b):
        np.negative(a,out=b)
        np.signbit(b,out=b)
    ##############################    
    # for Sigmoid    
    def SigmoidForward(self,a,b):
        # a : input
        # b : output location
        np.negative(a,out=b)
        np.exp(b,out=b)
        np.add(1,b,out=b)
        np.reciprocal(b,out=b)
        
    def SigmoidBackward(self,a,b):
        # a : sigmoid(x)
        # b : output location, compute >>> sigmoid(x)*(1-sigmoid(x))
        np.subtract(1,a,out=b)
        np.multiply(a,b,out=b)
    ##################################
