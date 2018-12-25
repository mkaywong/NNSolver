# NNSolver - Deep Neural Network Solver in python and numpy

In this project, I implemented a set of classes to support solving Deep Neural Networks. My goal is to explore the inner workings of different algorithms and enable one to gain insights into the computational complexity of solving and deploying Neural Networks. Currently, it supports building ResNet like networks. I have built an 18-layer network to perform image classification on MNIST data and achieved a 99.27% accuracy on the test set. For Cifar-10 dataset, I implemented a 24 layer ResNet and achieved a 76.18% accuracy on the test set. On the optimizer side, I implemented Momentum and Adam algorithms. 

# Getting Started

The download contains three folders.

* NNSolver - python classes
* Notebooks - a number of python notebooks to demonstrate how to run the networks
* Dataset - MNIST and Cifar-10 datasets. They are converted from the original datasets into .npz format.

# Prerequisites

All you need is python3, numpy and jupyter notebook.

# Module Description

The following is a brief description of the modules. Indendation shows the class inheritance.

* ### Utilities

  Defines ImageToArray and some plotting functios.

* ### Layer

  Base class for all Neural Network layers. It defines the basic interfaces to different layers. Hyper parameters such as learning rate, regularization etc. are defined as class variables. All instance layers will reference them during forward and backward computation. It also defines the different optimizers.

    * ### Linear 

      Defines common initialization functions for fully connected and 2D convolution layers

      * ### Conv2D 

        Defines 2D convolution functions. Note: Activation is not included in this layer.

      * ### FullyConnected

        Defines fully connected layer function. As in Conv2d, activation is not included in this layer.

    * ### PassThrough

      Defines common creation and initialization of data structures for layers that has the same input and output shapes.

      * ### Activation

        Currently supports ReLU and Sigmoid.

      * ### Normalize

        Normalizes the input with per channel mini batch mean and standard deviation.

      * ### Scale

        Scales the input with mean and standard deviation parameters calculated through optimization. Normalize followed by Scale together implement batch-norm function.

      * ### Pool

        Currently supports Max pool and Average pool.

      * ### Fork2

        Duplicates the input to form two branches.

      * ### Sum2

        Merges two branches by suming the respective inputs.

    * ### Cost

      Implements common interfaces for Softmax and MeanRSS layers.

      * ### Softmax

        Implements softmax functions.

      * ### MeanRSS

        Implements mean residual sum of squares.

    * ### Subnet

        Provides basic interface functions for a stack of layers.

      * ### ResNetBlock

        Defines the basec residual net layers.

      * ### Net

        Defines the required interface for the entire neural network.

        * ### RN24

          Defines an 24 layer residual network.

        * ### RN18 

          Defines an 18 layer residual network.

        * ### RN6
          
          Defines a 6 layer residual network.

    * ### ImageDataSource 

      Feeds the network with a mini-batch of samples on every forward computation.

* ### Solver

    Define the interfaces to perform optimization on the neural network with a given input source.
    
