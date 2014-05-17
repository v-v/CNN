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

		# calculate interval for random weights initialization
		l, h = self.act.sampleInterval(nPrev, nCurr)
	
		# initialize kernels to random values
		nCombinations = self.prevLayer.get_n() * self.currLayer.get_n()
		self.k = np.random.uniform(low = l, high = h, size = [nCombinations, self.kernelHeight, self.kernelWidth])

		# initialize one bias per feature map
		self.biasWeights = np.random.uniform(low = l, high = h, size = [self.currLayer.get_n()])
	
	def propagate(self):
		print "INPUT shape = ", self.currLayer.shape()
		self.FMs = np.zeros([self.currLayer.get_n(), self.currLayer.shape()[0], self.currLayer.shape()[1]])
		inFMs = self.prevLayer.get_x()
		print self.FMs

		k = 0 # kernel index, there is one foreach i, j combination
		for j in range(self.currLayer.get_n()): # foreach FM in the current layer
			for i in range(self.prevLayer.get_n()): # foreach FM in the previous layer
				if self.connections[i, j] == 1:
					print "\nprev FM", i, "is connected with FM ", j, "in current layer"

					# foreach neuron in the feature map
					for y_out in range(self.currLayer.shape()[0]):
						for x_out in range(self.currLayer.shape()[1]):

							# iterate inside the visual field for that neuron
							for y_k in range(0, self.kernelHeight, self.stepY):
								for x_k in range(0, self.kernelWidth, self.stepX):
									print i, "(", y_out + y_k, ",", x_out + x_k, ") -> ", j, "(", y_out, ",", x_out, ")"
									self.FMs[j, y_out, x_out] += inFMs[i, y_out + y_k, x_out + x_k] * self.k[k, y_k, x_k]
							# add bias
							self.FMs[j, y_out, x_out] += 1 * self.biasWeights[j]
				# next kernel
				k += 1

			# compute sigmoid (of a matrix since it's faster than elementwise)
			self.FMs[j] = self.act.func(self.FMs[j])

		print self.FMs





if __name__ == "__main__":

	in_data = np.array([ [
	      [[1, 1, 1, 0, 0, 0],
	       [1, 1, 1, 0, 0, 0],
	       [1, 1, 1, 0, 0, 0],
	       [1, 1, 1, 0, 0, 0],
	       [1, 1, 1, 0, 0, 0],
	       [1, 1, 1, 0, 0, 0]]
	     ] ])
	inLayer = layerFM(1, 6, 6, isInput = True)
	
	convLayer = layerFM(4, 1, 1)

	conv1 = convolutionalConnection(inLayer, convLayer, np.ones([1, 4]), 6, 6, 1, 1, )

	inLayer.set_x(in_data[0])
	conv1.propagate()
