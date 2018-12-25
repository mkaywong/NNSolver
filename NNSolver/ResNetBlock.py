#!/usr/bin/env python
# coding: utf-8

from Layer import Layer
from Subnet import Subnet
from Conv2D import Conv2D
from Activation import Activation
from Normalize import Normalize
from Scale import Scale
from Fork2 import Fork2
from Sum2 import Sum2

class ResNetBlock(Subnet):
    '''
    On the main path, the first convolution block has (1x1) kernel, zero padding. The second 
    convolution block has (3x3) kernel, (1,1) padding and stride of 1. The third convolution 
    block has (1x1) kernel, zero padding and stride of 1. The skip path has either identity mapping
    or a convolution block with (1x1) kernel and zero padding. The number of output channels is 
    the same as that of the third convolution block on the main path.
    
    Parameters required: 
    'instanceName': name of the block
    'skipMode': slect the operations on the skip path, 'conv' or 'identity'
    'skipStride': stride of the convolution block on the skip path
    'stride1': stride of the first convolution block on the main path
    'outChannel1': number of output channel of the first convolution block on the main path
    'outChannel2': number of output channel of the second convolution block on the main path
    'outChannel3': number of output channel of the third convolution block on the main path
    'activationType': activation function of the non-linear block, 'ReLU' or 'sigmoid'
    '''
    
    # 'conv' mode has a convolution block on the skip path. 'identity' mode is strict pass through.
    skipModes = ['conv', 'identity']
    
    def __init__(self,para):
        Subnet.__init__(self,para)
        self.layerList = []
        
        self.fork = Fork2({'instanceName':para['instanceName']+'_fork'})
        self.layerList.append(self.fork)
        self.skipMode = para['skipMode']
        if self.skipMode == 'conv':
            convPara4 = {'instanceName':para['instanceName']+'_skipConv1',
                     'padding':False, 
                     'padShape':(0,0),
                     'stride':para['skipStride'],
                     'outChannel': para['outChannel3'],
                     'kernelShape': (1,1),
                     'bias':False}
            self.skipConv = Conv2D(convPara4)
            self.skipNorm = Normalize({'instanceName':para['instanceName']+'_skipNorm'})
            self.skipScale = Scale({'instanceName':para['instanceName']+'_skipScale'})
            self.layerList.append(self.skipConv)
            self.layerList.append(self.skipNorm)
            self.layerList.append(self.skipScale)
            
        convPara1 = {'instanceName':para['instanceName']+'_mainConv1',
                     'padding':False, 
                     'padShape':(0,0),
                     'stride':para['stride1'],
                     'outChannel': para['outChannel1'],
                     'kernelShape': (1,1),
                     'bias':False}
        convPara2 = {'instanceName':para['instanceName']+'_mainConv2',
                     'padding':True, 
                     'padShape':(1,1),
                     'stride':1,
                     'outChannel': para['outChannel2'],
                     'kernelShape': (3,3),
                     'bias':False}
        convPara3 = {'instanceName':para['instanceName']+'_mainConv3',
                     'padding':False, 
                     'padShape':(0,0),
                     'stride':1,
                     'outChannel': para['outChannel3'],
                     'kernelShape': (1,1),
                     'bias':False}
        
        self.mainConv1 = Conv2D(convPara1)
        self.mainNorm1 = Normalize({'instanceName':para['instanceName']+'_mainNorm1'})
        self.mainScale1 = Scale({'instanceName':para['instanceName']+'_mainScale1'})
        self.mainActivation1 = Activation({'instanceName':para['instanceName']+'_mainReLU1',
                                      'activationType':para['activationType']})
        self.layerList.append(self.mainConv1)
        self.layerList.append(self.mainNorm1)
        self.layerList.append(self.mainScale1)
        self.layerList.append(self.mainActivation1)
        
        self.mainConv2 = Conv2D(convPara2)
        self.mainNorm2 = Normalize({'instanceName':para['instanceName']+'_mainNorm2'})
        self.mainScale2 = Scale({'instanceName':para['instanceName']+'_mainScale2'})
        self.mainActivation2 = Activation({'instanceName':para['instanceName']+'_mainReLU2',
                                      'activationType':para['activationType']})
        self.layerList.append(self.mainConv2)
        self.layerList.append(self.mainNorm2)
        self.layerList.append(self.mainScale2)
        self.layerList.append(self.mainActivation2)
        
        self.mainConv3 = Conv2D(convPara3)
        self.mainNorm3 = Normalize({'instanceName':para['instanceName']+'_mainNorm3'})
        self.mainScale3 = Scale({'instanceName':para['instanceName']+'_mainScale3'})
        self.layerList.append(self.mainConv3)
        self.layerList.append(self.mainNorm3)
        self.layerList.append(self.mainScale3)
          
        self.sum = Sum2({'instanceName':para['instanceName']+'_sum'})
        self.activation3 = Activation({'instanceName':para['instanceName']+'_outReLU3',
                                      'activationType':para['activationType']})
        self.layerList.append(self.sum)
        self.layerList.append(self.activation3)
        self.bottomInterface = self.fork
        self.topInterface = self.activation3
        
    def stack(self,top,bottom):
        self.top = top
        self.bottom = bottom
        if self.skipMode == 'conv':
            self.fork.fork(self.skipConv,self.mainConv1,bottom)
            self.skipConv.stack(self.skipNorm,self.fork.skip)
            self.skipNorm.stack(self.skipScale,self.skipConv)
            self.skipScale.stack(self.sum.skip,self.skipNorm)            
        else: 
            self.fork.fork(self.sum.skip,self.mainConv1,bottom)
        # main path
        self.mainConv1.stack(self.mainNorm1,self.fork.main)
        self.mainNorm1.stack(self.mainScale1,self.mainConv1)
        self.mainScale1.stack(self.mainActivation1,self.mainNorm1)
        self.mainActivation1.stack(self.mainConv2,self.mainScale1)
        
        self.mainConv2.stack(self.mainNorm2,self.mainActivation1)
        self.mainNorm2.stack(self.mainScale2,self.mainConv2)
        self.mainScale2.stack(self.mainActivation2,self.mainNorm2)
        self.mainActivation2.stack(self.mainConv3,self.mainScale2)
        
        self.mainConv3.stack(self.mainNorm3,self.mainActivation2)
        self.mainNorm3.stack(self.mainScale3,self.mainConv3)
        self.mainScale3.stack(self.sum.main,self.mainNorm3)
        # sum
        if self.skipMode == 'conv':
            self.sum.sum(self.activation3,self.skipScale,self.mainScale3)
        else:
            self.sum.sum(self.activation3,self.fork.skip,self.mainScale3)
        self.activation3.stack(top,self.sum)   
        