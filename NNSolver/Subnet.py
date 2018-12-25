#!/usr/bin/env python
# coding: utf-8

from Layer import Layer

class Subnet(Layer):
    '''
    Shell class for multiple layers. Inheritting classes need to define 
    __init__() and stack() methods. firstLayer and lastLayer also needs
    to be defined. 
    '''

    def __init__(self,para):
        Layer.__init__(self,para)
        self.layerList = []

    def createAndInitializeStruct(self):
        for i in range(len(self.layerList)):
            self.layerList[i].createAndInitializeStruct()
        self.out = self.topInterface.out

    def forward(self):
        for i in range(len(self.layerList)):
            self.layerList[i].forward()            
    
    def backward(self):
        for i in range(len(self.layerList)-1,-1,-1):
            self.layerList[i].backward()

    def updateParameters(self):
        for i in range(len(self.layerList)):
            self.layerList[i].updateParameters()
    
    def getParametersL2Sum(self):
        s = 0.
        for i in range(len(self.layerList)):
            s = s + self.layerList[i].getParametersL2Sum()
        return s

    def saveParameters(self, pD):
        for i in range(len(self.layerList)):
            self.layerList[i].saveParameters(pD)
    
    def loadParameters(self, pD):
        for i in range(len(self.layerList)):
            self.layerList[i].loadParameters(pD)

    def printIOShape(self):
        for i in range(len(self.layerList)):
            self.layerList[i].printIOShape()

    