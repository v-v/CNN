#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from utils import *
from featuremaps import *

# connects two layers by performing convolutions
class convolutionalConnection:
	def __init__(self, prevLayer, currLayer, connectionMatrix, kernelWidth, kernelHeight, stepX, stepY, useLogistic = False):
		self.connections  = connectionMatrix
		self.prevLayer   = prevLayer
		self.currLayer  = currLayer
		self.kernelHeight = kernelHeight
		self.kernelWidth  = kernelWidth
		self.stepX        = stepX
		self.stepY        = stepY

		if useLogistic:
			self.act = logistic()
		else:
			self.act = tanh()

		if prevLayer.get_n()  != np.shape(self.connections)[0] or \
		   currLayer.get_n() != np.shape(self.connections)[1]:
		   	print "Connection matrix size = ", self.connections.shape
			print "first layer = ", self.prevLayer.get_n()
			print "second layer = ", self.currLayer.get_n()
		   	raise Exception("convolutionalConnection: connection matrix shape does not match number" \
			"of feature maps in connecting layers")

		if np.ceil((self.prevLayer.get_x().shape[1] - self.kernelHeight) / self.stepY + 1) != self.currLayer.get_x().shape[1] or \
		   np.ceil((self.prevLayer.get_x().shape[2] - self.kernelWidth) / self.stepX + 1) != self.currLayer.get_x().shape[2]:
		   	raise Exception("Feature maps size mismatch")

		# random init kernels
		self.nKernels = self.prevLayer.get_n() * self.currLayer.get_n()
	
		# compute number of units in each layer (required to initlize weights)
		nPrev = self.prevLayer.get_n() * self.prevLayer.get_x().shape[1] * self.prevLayer.get_x().shape[2] 
		nCurr = self.currLayer.get_n() * self.currLayer.get_x().shape[1] * self.currLayer.get_x().shape[2]

		print "nPrev = ", nPrev
		print "nCurr = ", nCurr

		l, h = self.act.sampleInterval(nPrev, nCurr)
	
		nCombinations = self.prevLayer.get_n() * self.currLayer.get_n()
		self.k = np.random.uniform(low = l, high = h, size = [nCombinations, self.kernelHeight, self.kernelWidth])

		print self.k



if __name__ == "__main__":

	inLayer = layerFM(1, 6, 6, isInput = True)
	convLayer = layerFM(4, 1, 1)

	conv1 = convolutionalConnection(inLayer, convLayer, np.ones([1, 4]), 6, 6, 1, 1, )
