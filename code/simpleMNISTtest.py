#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from conv import *
from pooling import *
from mlp import *
from data import *
from sys import flush

if __name__ == "__main__":

	d = data()
	images, labels = d.loadData("../data/MNISTtrain-2-classes-norm.pkl")
	print "Loaded", len(images), "images of shape", images[0].shape

	inputLayer0  = layerFM(1, 32, 32, isInput = True) 
	convLayer1   = layerFM(6, 28, 28)
	poolLayer2   = layerFM(6, 14, 14)
	convLayer3   = layerFM(16, 10, 10)
	poolLayer4   = layerFM(16, 5, 5)
	convLayer5   = layerFM(20, 1, 1)
	hiddenLayer6 = layer1D(10)
	outputLayer7 = layer1D(2, isOutput = True)

	convolution01  = convolutionalConnection(inputLayer0, convLayer1, np.ones([1, 6]), 5, 5, 1, 1)	
	pooling12      = poolingConnection(convLayer1, poolLayer2, 2, 2)
	convolution23  = convolutionalConnection(poolLayer2, convLayer3, np.ones([6, 16]), 5, 5, 1, 1)	
	pooling34      = poolingConnection(convLayer3, poolLayer4, 2, 2)
	convolution45  = convolutionalConnection(poolLayer4, convLayer5, np.ones([16, 20]), 5, 5, 1, 1)
	full56         = fullConnection(convLayer5, hiddenLayer6)
	full67         = fullConnection(hiddenLayer6, outputLayer7)


	maxIter = 100
	eta = 0.001

	seq = np.r_[0:len(images)]
	for it in range(maxIter):
		np.random.shuffle(seq)

		total = 0.0
		correct = 0.0
		for i in range(len(images)):

			inputLayer0.set_FM(np.array([images[seq[i]]]))

			convolution01.propagate()
			pooling12.propagate()
			convolution23.propagate()
			pooling34.propagate()
			convolution45.propagate()
			full56.propagate()
			y = full67.propagate()

			full67.bprop(eta, labels[seq[i]])
			full56.bprop(eta)
			convolution45.bprop(eta)
			pooling34.bprop()
			convolution23.bprop(eta)
			pooling12.bprop()
			convolution01.bprop(eta)

			total += 1.0
			ok = False
			if np.argmax(y) == np.argmax(labels[seq[i]]):
				correct += 1.0
				ok = True
			if ok:
				print "Iteration", it, "sample", i, "(", labels[seq[i]], "), y =", y, "\t OK current TP =", correct / total
			else:
				print "Iteration", it, "sample", i, "(", labels[seq[i]], "), y =", y, "\t x  current TP =", correct / total

			sys.stdout.flush()

			if i % 100 == 0:
				# save weights, so we can continue
				f = gzip.open("weights-MNIST-2classes.pkl", 'wb')
				cPickle.dump((convolution01.k, convolution01.biasWeights, \
				              convolution23.k, convolution23.biasWeights, \
					      convolution45.k, convolution45.biasWeights, \
					      full56.w, full67.w), f)

				# generate kernel visualization
				for k in range(convolution01.k.shape[0]):
					plt.subplot(3, 2, k)
					plt.axis('off')
					#plt.imshow(convolution1.k[i], cmap=plt.cm.gray, interpolation='none')
					plt.imshow(convolution01.k[i], cmap=plt.cm.gray)

				plt.savefig("imgs/it"+str(it).zfill(2)+"_img"+str(i).zfill(5)+"_kernels.png")


		

