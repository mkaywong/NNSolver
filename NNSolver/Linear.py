#!/usr/bin/env python
# coding: utf-8
import numpy as np
import math
from Layer import Layer

class Linear(Layer):
    """
    Linear layer assumes there is a weight array W and a bias b.
    Covers Fully Connected Layers and Conv2D layers.
    """
    def __init__(self, Para):
        Layer.__init__(self,Para) 
        self.bias = False      # no bias term by default
        # The following names needs to be defined by inheriting layers
        self.outChannel = None
        self.W = None
        self.b = None
        self.dW = None
        self.db = None
        # for Momentum optimizer
        self.vdW = None
        self.vdb = None
        self.WShape = None    # used to initialize W
        self.inpNum = 0       # total number of input
        
    def isPadded(self):
        return self.padding
    
    def getPadShape(self):
        return self.padShape
        
    def setParameters(self, W, b=None):
        if self.W.shape != W.shape:
            print(self.W.shape, W.shape)
            self.printError("Weight W dimension does not match while setting layer parameters")
        np.copyto(self.W,W)
        if self.bias: 
            if self.b.shape != b.shape:
                print('self.b.shape:',self.b.shape, '!= b.shape', b.shape)
                self.printError("Bias b dimension does not match while setting layer parameters")
            else: np.copyto(self.b,b)

    def getParameters(self):
        return self.W, self.b
    
    def initParameters(self):
        # parameter initialization
        if Layer.initialization == "He-2015":
            self.He2015Init()
        elif Layer.initialization == 'Zero':
            self.zeroInit()
        elif Layer.initialization == 'One':
            self.oneInit()
        elif Layer.initialization == 'Random':
            self.randomInit()        
        self.dW = np.zeros(self.WShape,dtype=float)                         # dJ / dW        
        self.paraDict['W'] = self.W
        self.dParaDict['W'] = self.dW
        if self.bias: 
            self.db = np.zeros(self.outChannel,dtype=float)                  # dJ / db
            self.paraDict['b'] = self.b
            self.dParaDict['b'] = self.db
        
    def zeroInit(self):
        self.W = np.zeros(self.WShape,dtype=float)
        if self.bias: self.b = np.zeros(self.outChannel,dtype=float) 
            
    def oneInit(self):
        self.W = np.ones(self.WShape,dtype=float)
        if self.bias: self.b = np.zeros(self.outChannel,dtype=float)

    def randomInit(self):
        self.W = np.random.normal(size=self.WShape)
        if self.bias: self.b = np.zeros(self.outChannel,dtype=float)
        
    def He2015Init(self):        
        self.W = np.random.normal(loc=0.0, scale=math.sqrt(2.0/self.inpNum), size=self.WShape)
        if self.bias: self.b = np.zeros(self.outChannel,dtype=float)
    
    def printAll(self):
        print(self.instanceName)
        print('input:\n',self.bottom.getOutput())
        print('output:\n',self.out)
        print('Weight:\n',self.W)
        print('dW:\n',self.dW)
        if self.bias: 
            print('b:\n',self.b)
            print('db:\n',self.db)
        if Layer.optimizer == 'Momentum':
            print('vdW:\n',self.vdW)
            print('vdb:\n',self.vdb)
            