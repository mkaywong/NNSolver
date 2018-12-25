#!/usr/bin/env python
# coding: utf-8

import numpy as np
from Layer import Layer
import time

class Solver(object):

    miniCnt = 100

    def __init__(self,s,net):
        self.net = net
        self.source = s
    
    def createStruct(self):
        self.source.stack(self.net,None)
        self.net.stack(None,self.source)
        self.source.createAndInitializeStruct()
        self.net.createAndInitializeStruct()
        self.net.setSource(self.source)
    
    def saveParameters(self, fileName):
        pD = {}
        self.net.saveParameters(pD)
        np.savez(fileName,**pD)
    
    def loadParameters(self, fileName):
        pD = np.load(fileName)
        self.net.loadParameters(pD)

    def runForward(self):
        Layer.incrementMiniBatchCount()
        self.source.forward()
        self.net.forward()

    def runBackward(self):
        self.net.backward()
        self.source.backward()
    
    def runUpdate(self):
        self.net.updateParameters()

    def solveNMiniBatch(self,N):
        Layer.setInferenceMode(False)
        self.source.useTrainData()
        t1 = time.time()
        for i in range(N):
            if (i%Solver.miniCnt)==0: print(i,' ',end='')
            self.runForward()
            self.net.loss()
            self.runBackward()
            self.runUpdate()
        t2 = time.time()
        print("\nTime to process ",N," mini batches: ",t2-t1," seconds.")

    def solveNEpoch(self,N):
        Layer.setInferenceMode(False)
        self.source.useTrainData()
        n = 0
        i = 0
        t1 = time.time()
        while n != N:
            if (i % Solver.miniCnt)==0: print(i,' ',end='')
            self.runForward()
            self.net.loss()
            self.runBackward()
            self.runUpdate()
            i = i+1
            if self.source.endOfEpoch():
                n = n+1
                t3 = time.time()
                print('\nEpoch ', n, ' completed, time: ',t3-t1,' seconds')
        t2 = time.time()
        print("Time to process ",N," epoches: ",t2-t1," seconds.")

    def predict(self,source):
        Layer.setInferenceMode(True)
        if source == 'test':
            self.source.useTestData()
        elif source == 'train':
            self.source.useTrainData()
        done = False
        result = []
        i = 0
        t1 = time.time()
        while not done:
            if (i % Solver.miniCnt)==0: print(i,' ',end='')
            self.runForward()
            result.append(self.net.predict())
            i = i+1
            if self.source.endOfEpoch():
                done = True
        t2 = time.time()
        print()
        print("Time to predict ",source," data: ",t2-t1," seconds.")
        return result

    def calcAccuracy(self,source):
        '''
        Run one epoch on source
        '''
        Layer.setInferenceMode(True)
        if source == 'test':
            self.source.useTestData()
        elif source == 'train':
            self.source.useTrainData()
        done = False
        result = 0
        i = 0
        t1 = time.time()
        while not done:
            if (i % Solver.miniCnt)==0: print(i,' ',end='')
            self.runForward()
            result = result + self.net.calcAccuracy()
            if self.source.endOfEpoch():
                done = True
            i = i+1
        t2 = time.time()
        print()
        print("Time to calculate ",source," data accuracy: ",t2-t1," seconds.")
        print("Accuracy: ",result/self.source.currentDataSetSize(),' with data set size ',
                self.source.currentDataSetSize())
        return result

    def printIOShape(self):
        self.source.printIOShape()
        self.net.printIOShape()
