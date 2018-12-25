#!/usr/bin/env python
# coding: utf-8
import numpy as np
import sys
from Utilities import colors 

class Layer(object):
    """
    Base layer for Neural Networks.
    """
    inferenceMode = False
    batchSize = 1              # batch size for the run, and should be used to create the structures
    currentBatchSize = 1       # barch size that can be set by input layer to change the current batch
                               # used to take care of the case when sample size is not multiple of batch size
    miniBatchCount = 1
    layerList = []             # a list of layers, the order of which corresponds to the layers from 
                               # bottom to top. 
    debug = False

    # hyper parameters
    ##################################
    alpha = 0.01                # Learning rate    
    regularization = False      # set to True if regularization is used
    lmda = 0.001                # L2 regularization, weight decay
    # optimizer modes supported: Basic, Momentum, Adam
    optimizerList = ['Momentum', 'Adam', 'Basic']
    optimizer = "Basic"         # default optimizer is basic gradient descent
    betaM = 0.9                 # for 'Momentum'
    betaR = 0.999               # for rmsProp, not yet implemented. Both betaM and betaR are used in Adam
    epsilon = 0.00000001        # used in Adam to avoid dividing by zero
    
    initialization = "He-2015"  # Based on the paper by He-et-al, "Delving Deep into Rectifiers:
                                # Surpassing Human-Level Performance on ImageNet Classification"
                                # Other modes supported 'Zeros', 'Ones' for debuging
        
    def __init__(self, para):
        # Para is a dictionary of the hyper parameters used in the layer. 
        self.hPara = para
        self.out = None          # Layer activations
        self.pOut = None         # activation padded with zeros with the top layer size
        self.inpDeriv = None      # computed partial derivatives of input (bottom layer output) for backprop
        self.pInpDeriv = None     # padded version of inpDeriv, if the bottom layer is padded       
        self.padding = False   # default is no padding
        self.padShape = (0,0)   # a 2D tuple

        self.paraDict = {}      # a dictionary of parameters, used during initialization and updates
        self.dParaDict = {}     # a dictionary for the derivatives of the parameter, has to share the same
                                # keys with paraDict
        self.top = None
        self.bottom = None
        self.topInterface = self
        self.bottomInterface = self
        
        try:
            self.instanceName = para['instanceName']          
        except:
            self.printWarning("No 'instanceName' specified.")
            self.instanceName = None
    
    def __str__(self):
        s = colors.BOLD+type(self).__name__+colors.END+' - '+self.instanceName+' '
        return s
    
    def getOutput(self):
        # Used in the forward pass, top layer retrive activation from bottom layer
        return self.topInterface.out
    
    def getPaddedOutput(self):
        return self.topInterface.pOut
    
    def getInputDerivative(self):
        #used in the backward pass, bottom layer retrive derivative with respect to activation from top layer
        return self.bottomInterface.inpDeriv
    
    def getPInputDerivative(self):
        return self.bottomInterface.pInpDeriv
    
    def stack(self, top, bottom):
        if top != None:
            self.top = top.bottomInterface
        else: 
            self.top = None
        if bottom != None:
            self.bottom = bottom.topInterface
        else:
            self.bottom = None
    
    def createAndInitializeStruct(self):
        return
    
    def forward(self):
        return
    
    def backward(self):
        return
    
    def getParametersL2Sum(self):
        s = 0.
        for k in self.paraDict.keys():
            s = s + np.sum(np.square(self.paraDict[k]))
        return s
    
    def saveParameters(self, pD):
        for k in self.paraDict.keys():
            pD[self.instanceName+':'+k] = self.paraDict[k]
        
    def loadParameters(self, pD):
        try:
            for k in self.paraDict.keys():
                np.copyto(self.paraDict[k],pD[self.instanceName+':'+k])
        except:
            self.printError('Layer parameter(s) does not exist while loading.')

    def printError(self,s):
        print(colors.BOLD+"Error:\n"+colors.END+self.__str__()+"\n"+s)
        sys.exit()

    def printWarning(self,s):
        print(colors.BOLD+"Warning:\n"+colors.END+self.__str__()+"\n"+s) 
        
    def isPadded(self):
        return self.padding
    
    def printDebugForward(self):
        if Layer.debug:
            print(self.instanceName, ': output')
            print(self.out)
    
    def printDebugBackward(self):
        if Layer.debug:
            print(self.instanceName, ': input Derivative')
            print(self.inpDeriv)
    
    def printIOShape(self):
        print(colors.BOLD+self.instanceName+colors.END+' input: ',end="")           
        if self.bottom != None:
            print(self.bottom,' ',self.bottom.getOutput().shape,end="")
        else: 
            print("None",end="")
        print(' output:',self.out.shape)
        for k in self.paraDict.keys():
            print('\t'+k+':',self.paraDict[k].shape)

    #
    # Optimization 
    #
    def createOptimizerStruct(self):        
        # optimizers that have additional structures that need to be created and initialized
        if Layer.optimizer == "Momentum":
            self.momentumCreateStruct()
        if Layer.optimizer == "Adam":
            self.adamCreateStruct()
                    
    def momentumCreateStruct(self):      
        self.mStruct = {}
        for k in self.paraDict.keys():
            self.mStruct[k] = np.zeros(self.paraDict[k].shape,dtype=float)                      
    
    def adamCreateStruct(self):      
        self.mStruct = {}         # Momentum
        self.mStructC = {}        # bias corrected version of momentum
        self.rStruct = {}         # RMS
        self.rStructC = {}        # bias corrected version RMS
        for k in self.paraDict.keys():
            self.mStruct[k] = np.zeros(self.paraDict[k].shape,dtype=float)
            self.mStructC[k] = np.zeros(self.paraDict[k].shape,dtype=float)
            self.rStruct[k] = np.zeros(self.paraDict[k].shape,dtype=float)
            self.rStructC[k] = np.zeros(self.paraDict[k].shape,dtype=float)                      
    
    def updateParameters(self):
        if Layer.optimizer == "Momentum":
            self.momentumUpdate()
        elif Layer.optimizer == 'Adam':
            self.adamUpdate()
        elif Layer.optimizer == 'Basic':
            self.basicUpdate()
        
    def adamUpdate(self):
        mB = (1-Layer.betaM**Layer.miniBatchCount)        # bias adjustment for momentum 
        rB = (1-Layer.betaR**Layer.miniBatchCount)        # bias adjustment for rmsProp
        for k in self.paraDict.keys():
            np.add(Layer.betaM*self.mStruct[k],(1-Layer.betaM)*self.dParaDict[k],
                  out = self.mStruct[k])
            np.divide(self.mStruct[k],mB,out=self.mStructC[k])
            
            np.add(Layer.betaR*self.rStruct[k],(1-Layer.betaR)*np.square(self.dParaDict[k]),
                  out = self.rStruct[k])
            np.divide(self.rStruct[k],rB,out=self.rStructC[k])
            
            np.sqrt(self.rStructC[k],out=self.rStructC[k])
            np.add(self.rStructC[k],Layer.epsilon,out=self.rStructC[k])            
            np.divide(self.mStructC[k],self.rStructC[k],out=self.rStructC[k])
            np.subtract(self.paraDict[k],Layer.alpha*self.rStructC[k],
                       out = self.paraDict[k])
        
    def momentumUpdate(self): 
        for k in self.paraDict.keys():
            np.add(Layer.betaM*self.mStruct[k],(1-Layer.betaM)*self.dParaDict[k],
                  out = self.mStruct[k])
            np.subtract(self.paraDict[k],Layer.alpha*self.mStruct[k],
                       out = self.paraDict[k])
        
    def basicUpdate(self):
        for k in self.paraDict.keys():
            np.subtract(self.paraDict[k], Layer.alpha*self.dParaDict[k], out = self.paraDict[k])
    
    @staticmethod
    def incrementMiniBatchCount():
        Layer.miniBatchCount = Layer.miniBatchCount+1

    @staticmethod
    def setInferenceMode(mode):
        Layer.inferenceMode = mode

    @staticmethod
    def setHyperParameters(hPara):
        Layer.batchSize = hPara['BATCH_SIZE']
        Layer.currentBatchSize = Layer.batchSize
        Layer.alpha = hPara['ALPHA']
        Layer.regularization = hPara['REGULARIZATION']
        if Layer.regularization:
            Layer.lmda = hPara['LAMBDA']
        Layer.optimizer = hPara['OPTIMIZER'] 
        if Layer.optimizer == 'Momentum':
            Layer.betaM = hPara['BETAM']
        if Layer.optimizer == 'Adam':
            Layer.betaM = hPara['BETAM']
            Layer.betaR = hPara['BETAR']
        Layer.initialization = hPara['INITIALIZATION']

    @staticmethod
    def updateLearningRate(alpha):
        Layer.alpha = alpha