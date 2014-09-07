#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Convolutional Neural Network library
# copyright (c) 2014 Vedran Vukotic
# gmail: vevukotic

# visualizeKernelsMASTIF.py
# loads a learnt model from model/weights-MASTIF.py
# and generates some visualizations of the learnt filters

import numpy as np
from conv import *
from pooling import *
from mlp import *
from data import *
import sys

inputLayer0  = layerFM(3, 48, 48, isInput = True) 
convLayer1   = layerFM(10, 42, 42)
poolLayer2   = layerFM(10, 21, 21)
convLayer3   = layerFM(15, 18, 18)
poolLayer4   = layerFM(15, 9, 9)
convLayer5   = layerFM(20, 6, 6)
poolLayer6   = layerFM(20, 3, 3)
convLayer7   = layerFM(40, 1, 1)
hiddenLayer8 = layer1D(80)
outputLayer9 = layer1D(9, isOutput = True)

convolution01  = convolutionalConnection(inputLayer0, convLayer1, np.ones([3, 10]), 7, 7, 1, 1)	
pooling12      = poolingConnection(convLayer1, poolLayer2, 2, 2)
convolution23  = convolutionalConnection(poolLayer2, convLayer3, np.ones([10, 15]), 4, 4, 1, 1)	
pooling34      = poolingConnection(convLayer3, poolLayer4, 2, 2)
convolution45  = convolutionalConnection(poolLayer4, convLayer5, np.ones([15, 20]), 4, 4, 1, 1)
pooling56      = poolingConnection(convLayer5, poolLayer6, 2, 2)
convolution67  = convolutionalConnection(poolLayer6, convLayer7, np.ones([20, 40]), 3, 3, 1, 1)
full78         = fullConnection(convLayer7, hiddenLayer8)
full89         = fullConnection(hiddenLayer8, outputLayer9)

f = gzip.open("../models/weights-MASTIF.pkl")
(convolution01.k, convolution01.biasWeights, \
 convolution23.k, convolution23.biasWeights, \
 convolution45.k, convolution45.biasWeights, \
 convolution67.k, convolution67.biasWeights, \
 full78.w, full89.w) = cPickle.load(f)
f.close()

#fig = plt.figure(num=None, figsize=(10, 40), dpi=80, facecolor='w', edgecolor='k')

for i in range(30):
	plt.subplot(5, 6, i+1)
	plt.axis('off')
	plt.imshow(convolution01.k[i], cmap=plt.cm.gray)
	#plt.title("i("+str(c)+", b)="+acceptedClasses[np.argmax(labels[c])])
	

plt.savefig("kernels_MASTIF_Layer1.png")

plt.clf()
for i in range(150):
	plt.subplot(15, 10, i+1)
	plt.axis('off')
	plt.imshow(convolution23.k[i], cmap=plt.cm.gray)
	#plt.title("i("+str(c)+", b)="+acceptedClasses[np.argmax(labels[c])])
	

plt.savefig("kernels_MASTIF_Layer2.png")
