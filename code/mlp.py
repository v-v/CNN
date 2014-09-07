#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Convolutional Neural Network library
# copyright (c) 2014 Vedran Vukotic
# gmail: vevukotic

# mlp.py - implements a MultiLayer perceptron


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
	def sampleMSE(self, expectedOutputs):
		if not self.isOutput: raise Exception("MSE should only be computed on output neurons")
		self.dErrors = np.sum( (self.x - expectedOutputs) ** 2.0) / 2.0
		return self.dErrors
	
	# set layer error
	def set_error(self, deriv):
		self.error = deriv

	# get layer error
	def get_error(self):
		return self.error


# fullConnection
# represents a full connection between two
# layers in a fully connected Neural Network
# and implements forward and backpropagation
# between two fully connected layers

# By default a tanh activation fuction is used
# (as it's slightly faster). To use a Logistic
# Sigmoid function set useLogistic True
class fullConnection:
	def __init__(self, prevLayer, currLayer, useLogistic = False, w = None):
		self.prevLayer = prevLayer
		self.currLayer = currLayer

		if useLogistic:
			self.act = logistic()
		else:
			self.act = tanh()

		self.nPrev = prevLayer.get_size()
		self.nCurr = currLayer.get_size()

		l, h = self.act.sampleInterval(self.nPrev, self.nCurr)

		# if the current layer has a bias
		# we add another weight pointing to a permanent 1
		if currLayer.hasBias():
			self.nPrev += 1

		if w is None:
			self.w = np.random.uniform(low = l, high = h, size = [self.nPrev, self.nCurr])
		else:
			self.w = w

		return None
	
	def propagate(self):
		x = self.prevLayer.get_x()[np.newaxis]
		if self.currLayer.hasBias:
			x = np.append(x, [1])

		z = np.dot(self.w.T, x)
		
		# compute and store output
		y = self.act.func(z)
		self.currLayer.set_x(y)

		return y
	
	def bprop(self, ni, target = None, verbose = False):
		yj = self.currLayer.get_x()
		if verbose: 
			print "out = ", yj
			print "w = ", self.w

		# compute or retreive error of current layer
		if self.currLayer.isOutput:
			if target is None: raise Exception("bprop(): target values needed for output layer")
			currErr = -(target - yj) * self.act.deriv(yj)
			self.currLayer.set_error(currErr)
		else:
			currErr = self.currLayer.get_error()

		if verbose: print "currErr =  ", currErr

		yi = np.append(self.prevLayer.get_x(), [1])
		# compute error of previous layer
		if not self.prevLayer.isInput:
			prevErr = np.zeros(len(yi))
			for i in range(len(yi)):
				prevErr[i] = sum(currErr * self.w[i]) * self.act.deriv(yi[i])

			self.prevLayer.set_error(np.delete(prevErr,-1))

		# compute weight updates
		dw = np.dot(np.array(yi)[np.newaxis].T, np.array(currErr)[np.newaxis])
		self.w -= ni * dw
	
	def get_weights(self):
		return self.w
	
	def set_weights(self, w):
		self.w = w

