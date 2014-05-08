#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from utils import *

# layer1D
# represents a 1D layer, this can be:
# - a 1D input layer (not used in CNNs since we work with images that are 2D)
# - an output layer
# - any hidden layer that's part of a fully connected NN
class layer1D:
	def __init__(self, n, isInput = False, isOutput = False, hasBias = None, x = None):
		self.n = n
		self.isInput = isInput
		self.isOutput = isOutput
		if isInput:
			self.withBias = False
		else:
			self.withBias = True
		if hasBias is not None:
			self.withBias = hasBias
		
		if x is None:
			self.x = np.zeros(n)
		else:
			if len(x) != self.n: raise Exception("Input (x) size should should be equal to n ("+str(n)+")")
			self.x = x
		
		self.b = None
		if hasBias:
			b = 1

		if isInput and isOutput: raise Exception("Neuron Layer can't be both an input layer and an output one")

	def set_x(self, x):
		if len(x) != self.n: raise Exception("Input (x) size should should be equal to n ("+str(n)+")")
		self.x = x

	def get_x(self):
		return self.x
	
	# returns number of neurons in layer
	def get_size(self):
		return self.n
	
	def hasBias(self):
		return self.withBias
	
	# compute MSE for an output layer
	def MSE(self, expectedOutputs):
		if not isOutput: raise Exception("MSE should only be computed on output neurons")
		return np.sum(self.w - expectedOutputs) / 2.0

# fullConnection
# represents a full connection between two
# layers in a fully connected Neural Network
# and implements forward and backpropagation
# between two fully connected layers

# By default a tanh activation fuction is used
# (as it's slightly faster). To use a Logistic
# Sigmoid function set useLogistic True
class fullConnection:
	def __init__(self, prevLayer, currLayer, useLogistic = False):
		self.prevLayer = prevLayer
		self.currLayer = currLayer

		if useLogistic:
			self.act = logistic()
		else:
			self.act = tanh()

		l, h = self.act.sampleInterval(prevLayer.get_size(), currLayer.get_size())

		self.nPrev = prevLayer.get_size()
		self.nCurr = currLayer.get_size()

		# if the current layer has a bias
		# we add another weight to a permanent 1
		if currLayer.hasBias():
			self.nPrev += 1

		self.w = np.random.uniform(low = l, high = h, size = [self.nCurr, self.nPrev])

		print "prevLayer = ", prevLayer.get_size()
		print "currLayer = ", currLayer.get_size()
		print "l = ", l, "h = ", h

		print self.w
		return None
	
	def propagate(self, bprop = False):
		x = self.prevLayer.get_x()[np.newaxis]
		if self.currLayer.hasBias:
			x = np.append(x, [1])

		print "x = ", x
		z = np.dot(x, self.w.T)
		print "z = ", z
		y = self.act.func(z)
		print "y = ", y
		self.currLayer.set_x(y)

		return None

if __name__ == "__main__":		
	
	layer0 = layer1D(2, isInput = True, x = np.array([0, 1]))
	layer1 = layer1D(5)
	layer2 = layer1D(1, isOutput = True)

	print "Subnet in -> hidden "
	print "===================="

	subnet01 = fullConnection(layer0, layer1)

	print ""
	print "Subnet hidden -> out"
	print "===================="
	subnet12 = fullConnection(layer1, layer2)

	print "\nIN -> HIDDEN"
	subnet01.propagate()

	print "\nHIDDEN -> OUT"
	subnet12.propagate()
