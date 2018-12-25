#!/usr/bin/env python
# coding: utf-8

from Subnet import Subnet

class Net(Subnet):
    '''
    Net class extends the Subnet class by adding the interface to the last
    layer, where loss, accuracy are calculated. The Net layer needs to be
    set to the last layer (via setNet method) as well.
    '''

    def setSource(self, s):
        self.layerList[-1].setSource(s)

    def loss(self):
        self.layerList[-1].loss()
    
    def lossData(self):
        return self.layerList[-1].lossData()

    def predict(self):
        return self.layerList[-1].predict()
    
    def calcAccuracy(self):
        return self.layerList[-1].calcAccuracy()
    