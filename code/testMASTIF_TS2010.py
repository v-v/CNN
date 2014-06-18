#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from conv import *
from pooling import *
from mlp import *
from data import *
import sys

if __name__ == "__main__":

	d = data()
	images, labels = d.loadData("../data/mastif_ts2010_test.pkl")
	print "Loaded", len(images), "images of shape", images[0][0].shape, "with", len(images[0]), "channels"

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

	f = gzip.open("weights-MASTIF_simplified.pkl")
	(convolution01.k, convolution01.biasWeights, \
	 convolution23.k, convolution23.biasWeights, \
	 convolution45.k, convolution45.biasWeights, \
	 convolution67.k, convolution67.biasWeights, \
	 full78.w, full89.w) = cPickle.load(f)
	f.close()

	seq = np.r_[0:len(images)]
	np.random.shuffle(seq)

	total = 0.0
	correct = 0.0
	for i in range(len(images)):

		inputLayer0.set_FM(np.array(images[seq[i]]))

		convolution01.propagate()
		pooling12.propagate()
		convolution23.propagate()
		pooling34.propagate()
		convolution45.propagate()
		pooling56.propagate()
		convolution67.propagate()
		full78.propagate()
		y = full89.propagate()

		total += 1.0
		ok = False
		np.set_printoptions(precision=3)
		if np.argmax(y) == np.argmax(labels[seq[i]]):
			correct += 1.0
			ok = True
		if ok:
			print "sample", i, "(", labels[seq[i]], "), y =", y, "\t OK current TP =", correct / total
		else:
			print "sample", i, "(", labels[seq[i]], "), y =", y, "\t x  current TP =", correct / total

		sys.stdout.flush()


		

