#!/usr/bin/env python
# coding: utf-8

from Net import Net
from FullyConnected import FullyConnected
from Conv2D import Conv2D
from Activation import Activation
from Softmax import Softmax
from Normalize import Normalize
from Scale import Scale
from Pool import Pool
from ResNetBlock import ResNetBlock

class RN6(Net):
    '''
    ResNet6 has a total of 6 layers: 
    Note that some parameters are predetermined. The parameters need to be specified are in ''.
    For all ResNetBlock modules, the output sizes of stage 1 and stage 2 conv2D blocks equals to 
    1/4 of that of the final stage.
    Conv1 - kernel:(3x3), pad:(1,1), stride:1, output: 'c1OutChannel'
    Conv1 - kernel:(3x3), pad:(1,1), stride:2, output: 'c2OutChannel'  # H and W reduced by half
    RNB1 - skipMode:conv, skipStride:2, output : 'rnb1OutChannel' # H and W reduced by half
    pool - average pooling of RNB5 per channel, reducing output to 'rnb1OutChannel', need to specify
            'pSize', which is used to specify stride and kernel size
    fc - outChannel: 'classNum'
    softmax - final classification layer
    '''
    def __init__(self,para):
        Net.__init__(self,para)
        convPara1 = {'instanceName':'RN6'+'_Conv1',
            'padding':True, 
            'padShape':(1,1),
            'stride':1,
            'outChannel': para['c1OutChannel'],
            'kernelShape': (3,3),
            'bias':False}
        self.conv1 = Conv2D(convPara1)
        self.norm1 = Normalize({'instanceName':'RN6'+'_Norm1'})
        self.scale1 = Scale({'instanceName':'RN6'+'_Scale1'})
        self.activation1 = Activation({'instanceName':'RN6'+'_Activation1',
                                'activationType':'ReLU'})
        self.layerList.append(self.conv1)
        self.layerList.append(self.norm1)
        self.layerList.append(self.scale1)
        self.layerList.append(self.activation1)
        convPara2 = {'instanceName':'RN6'+'_Conv2',
            'padding':True, 
            'padShape':(1,1),
            'stride':2,
            'outChannel': para['c2OutChannel'],
            'kernelShape': (3,3),
            'bias':False}
        self.conv2 = Conv2D(convPara2)
        self.norm2 = Normalize({'instanceName':'RN6'+'_Norm2'})
        self.scale2 = Scale({'instanceName':'RN6'+'_Scale2'})
        self.activation2 = Activation({'instanceName':'RN6'+'_Activation2',
                                'activationType':'ReLU'})
        self.layerList.append(self.conv2)
        self.layerList.append(self.norm2)
        self.layerList.append(self.scale2)
        self.layerList.append(self.activation2)
        self.rnb1 = ResNetBlock({'instanceName':'RN6'+'_RNB1', 
            'skipMode': 'conv', 
            'skipStride': 2,
            'stride1': 2,
            'outChannel1': int(para['rnb1OutChannel']/4),
            'outChannel2': int(para['rnb1OutChannel']/4),
            'outChannel3': para['rnb1OutChannel'],
            'activationType': 'ReLU' 
             })
        self.layerList.append(self.rnb1)
        self.pool1 = Pool({'instanceName':'RN6'+'_pool1', 'poolType':'ave', 
            'stride':para['pSize'], 'kernelShape':(para['pSize'],para['pSize'])})
        self.layerList.append(self.pool1)
        self.fc1 = FullyConnected({'instanceName':'RN6'+'_fc1','outChannel':para['classNum'],
            'bias':True})
        self.layerList.append(self.fc1)
        self.softmax = Softmax({'instanceName':'RN6'+'_softmax'})
        self.layerList.append(self.softmax)
        self.bottomInterface = self.conv1
        self.topInterface = self.softmax 
        self.softmax.setNet(self)

    def stack(self,top,bottom):
        self.top = top
        self.bottom = bottom
        self.conv1.stack(self.norm1,bottom)
        self.norm1.stack(self.scale1,self.conv1)
        self.scale1.stack(self.activation1,self.norm1)
        self.activation1.stack(self.conv2,self.scale1)

        self.conv2.stack(self.norm2,self.activation1)
        self.norm2.stack(self.scale2,self.conv2)
        self.scale2.stack(self.activation2,self.norm2)
        self.activation2.stack(self.rnb1,self.scale2)

        self.rnb1.stack(self.pool1,self.activation2)
        self.pool1.stack(self.fc1,self.rnb1)
        self.fc1.stack(self.softmax,self.pool1)
        self.softmax.stack(top,self.fc1) 
        self.softmax.setSource(bottom)  
