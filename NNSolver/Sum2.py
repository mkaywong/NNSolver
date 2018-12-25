#!/usr/bin/env python
# coding: utf-8

import numpy as np
from Utilities import colors
from Layer import Layer
from PassThrough import PassThrough

class Sum2(PassThrough):
    def __init__(self,para):
        PassThrough.__init__(self,para)
        self.skip = PassThrough({'instanceName':para['instanceName']+'_skip'})
        self.main = PassThrough({'instanceName':para['instanceName']+'_main'})
    
    def sum(self,top, botSkip,botMain):
        self.skip.stack(top,botSkip)
        self.main.stack(top,botMain)
        self.top = top.bottomInterface
        # set the bottom to the main path. for collecting dimension only
        self.bottom = self.main.bottom
    
    def createAndInitializeStruct(self):
        self.skip.createAndInitializeBotStruct()
        self.main.createAndInitializeBotStruct()
        # check the skip and main path input dimensions 
        if self.skip.bottom.getOutput().shape != self.main.bottom.getOutput().shape:
            self.printError("Skip and main input dimensions do not match.")            
        self.createAndInitializeTopStruct()
        
    def forward(self):
        cbs = Layer.currentBatchSize
        np.add(self.skip.bottom.getOutput()[:cbs],self.main.bottom.getOutput()[:cbs],
               out=self.out[:cbs])  
    
    def backward(self):
        cbs = Layer.currentBatchSize 
        np.copyto(self.main.inpDeriv[:cbs],self.top.getInputDerivative()[:cbs])
        np.copyto(self.skip.inpDeriv[:cbs],self.top.getInputDerivative()[:cbs])             
    
    def printIOShape(self):
        print(colors.BOLD+self.instanceName+colors.END,end='')
        print(' skip:',self.skip.bottom,' ',self.skip.bottom.getOutput().shape,end='')
        print(' main:',self.main.bottom,' ',self.main.bottom.getOutput().shape,end='')
        print(' output:',self.out.shape)

