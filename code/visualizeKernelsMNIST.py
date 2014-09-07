#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Convolutional Neural Network library
# copyright (c) 2014 Vedran Vukotic
# gmail: vevukotic

# visualizeKernelsMNIST.py
# loads a learnt model from model/weights-MNIST.py
# and generates visualizations of the learnt filters

import numpy as np
from conv import *
from pooling import *
from mlp import *
from data import *
import sys

inputLayer0  = layerFM(1, 32, 32, isInput = True) 
convLayer1   = layerFM(6, 28, 28)
poolLayer2   = layerFM(6, 14, 14)
convLayer3   = layerFM(16, 10, 10)
poolLayer4   = layerFM(16, 5, 5)
convLayer5   = layerFM(100, 1, 1)
hiddenLayer6 = layer1D(80)
outputLayer7 = layer1D(10, isOutput = True)

convolution01  = convolutionalConnection(inputLayer0, convLayer1, np.ones([1, 6]), 5, 5, 1, 1)	
pooling12      = poolingConnection(convLayer1, poolLayer2, 2, 2)
convolution23  = convolutionalConnection(poolLayer2, convLayer3, np.ones([6, 16]), 5, 5, 1, 1)	
pooling34      = poolingConnection(convLayer3, poolLayer4, 2, 2)
convolution45  = convolutionalConnection(poolLayer4, convLayer5, np.ones([16, 100]), 5, 5, 1, 1)
full56         = fullConnection(convLayer5, hiddenLayer6)
full67         = fullConnection(hiddenLayer6, outputLayer7)

f = gzip.open("../models/weights-MNIST.pkl")
(convolution01.k, convolution01.biasWeights, \
 convolution23.k, convolution23.biasWeights, \
 convolution45.k, convolution45.biasWeights, \
 full56.w, full67.w) = cPickle.load(f)
f.close()



for i in range(6):
	plt.subplot(2, 3, i+1)
	plt.axis('off')
	plt.imshow(convolution01.k[i], cmap=plt.cm.gray)
	#plt.title("i("+str(c)+", b)="+acceptedClasses[np.argmax(labels[c])])
	

plt.savefig("kernels_MNIST_Layer1.png")

plt.clf()
for i in range(96):
	plt.subplot(12, 8, i+1)
	plt.axis('off')
	plt.imshow(convolution23.k[i], cmap=plt.cm.gray)
	#plt.title("i("+str(c)+", b)="+acceptedClasses[np.argmax(labels[c])])
	

plt.savefig("kernels_MNIST_Layer2.png")
