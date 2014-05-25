#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from conv import *
from pooling import *
from mlp import *
from data import *
import sys

if __name__ == "__main__":



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


	f = gzip.open("weights-MNIST.pkl")
	(convolution01.k, convolution01.biasWeights, \
	 convolution23.k, convolution23.biasWeights, \
	 convolution45.k, convolution45.biasWeights, \
	 full56.w, full67.w) = cPickle.load(f)
	f.close()

	d = data()
	images, labels = d.loadData("../data/MNISTtest-norm.pkl")
	print "Loaded", len(images), "images of shape", images[0].shape


	total = 0.0
	correct = 0.0
	for i in range(len(images)):

		inputLayer0.set_FM(np.array([images[i]]))

		convolution01.propagate()
		pooling12.propagate()
		convolution23.propagate()
		pooling34.propagate()
		convolution45.propagate()
		full56.propagate()
		y = full67.propagate()

		total += 1.0
		ok = False
		if np.argmax(y) == np.argmax(labels[i]):
			correct += 1.0
			ok = True
		if ok:
			print "Sample", i, "(", labels[i], "), y =", y, "\t OK current TP =", correct / total
		else:
			print "Sample", i, "(", labels[i], "), y =", y, "\t x  current TP =", correct / total

		sys.stdout.flush()


