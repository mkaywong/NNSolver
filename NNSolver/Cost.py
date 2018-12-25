#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math
from Layer import Layer
from PassThrough import PassThrough

class Cost(PassThrough):
    """
    Cost Layer: provide interfaces to compute the cost (loss) of the (mini-)batch. 
    """
    axisDim = [(1),(1,2),(1,2,3),(1,2,3,4)]
    def createAndInitializeStruct(self):
        PassThrough.createAndInitializeStruct(self) 
        self.lossOut = []

    def setSource(self,s):
        '''
        s: Source layer to retrieve the labels associated with the current batch
        '''
        self.source = s
    
    def setNet(self,n):
        '''
        n: the network the softmax layer is in; used in calculating the L2 sum of the parameters
           of all the layers in the net
        '''
        self.net = n
    
    def lossData(self):
        '''
        Return a list of loss computed so far
        '''
        return self.lossOut

#
# The following needs to be defined by the inheriting class, in addition to forward and backward
#                
    def loss(self):
        """
        Return the mini-batch loss. Need to be defined
        """
        return
        
    def predict(self):
        '''
        Return the class numbers or regression values for the mini-batch
        '''
        return 

    def calcAccuracy(self):
        """
        Return the number of correct classification or the error of the regression for the mini-batch
        """
        return
