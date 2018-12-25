#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math
from Utilities import colors
from Layer import Layer
from PassThrough import PassThrough

class Fork2(PassThrough):
    
    def __init__(self,para):
        PassThrough.__init__(self,para)
        self.skip = PassThrough({'instanceName':para['instanceName']+'_skip'})
        self.main = PassThrough({'instanceName':para['instanceName']+'_main'})
        
    def fork(self,topSkip,topMain,bot):
        self.skip.stack(topSkip,bot)
        self.main.stack(topMain,bot)
        self.bottom = bot.topInterface
    
    def createAndInitializeStruct(self):
        self.skip.createAndInitializeTopStruct()
        self.main.createAndInitializeTopStruct()
        self.createAndInitializeBotStruct()
        
    def forward(self):
        cbs = Layer.currentBatchSize
        np.copyto(self.skip.out[:cbs],self.bottom.getOutput()[:cbs])
        np.copyto(self.main.out[:cbs],self.bottom.getOutput()[:cbs])
    
    def backward(self):
        cbs = Layer.currentBatchSize
        np.add(self.skip.top.getInputDerivative()[:cbs],self.main.top.getInputDerivative()[:cbs],
               out=self.inpDeriv[:cbs])
    
    def printIOShape(self):       
        print(colors.BOLD+self.instanceName+colors.END,end='')
        print(' input:',self.bottom,' ',self.bottom.getOutput().shape,end='')
        print(' skip:',self.skip.getOutput().shape,end='')
        print(' main:',self.main.getOutput().shape)
        