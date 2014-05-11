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
		if not self.isOutput: raise Exception("MSE should only be computed on output neurons")
		self.dErrors = np.sum( (self.x - expectedOutputs) ** 2.0) / 2.0
		return self.dErrors
	
	# set derivative
	def set_deriv(self, deriv):
		self.deriv = deriv
	
	def get_deriv(self):
		return self.deriv

	# get error derivatives
	def get_dErr(self):
		return self.dErr

	# set error derivatives
	def set_dErr(self, dErr):
		self.dErr = dErr
		

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

		l, h = self.act.sampleInterval(prevLayer.get_size(), currLayer.get_size())

		self.nPrev = prevLayer.get_size()
		self.nCurr = currLayer.get_size()

		# if the current layer has a bias
		# we add another weight to a permanent 1
		if currLayer.hasBias():
			self.nPrev += 1

#		self.w = np.random.uniform(low = l, high = h, size = [self.nCurr, self.nPrev])
		if w is None:
			self.w = np.random.uniform(low = l, high = h, size = [self.nPrev, self.nCurr])
		else:
			self.w = w

#		print "prevLayer = ", prevLayer.get_size()
#		print "currLayer = ", currLayer.get_size()
#		print "l = ", l, "h = ", h

		#print self.w
		return None
	
	def propagate(self):
		x = self.prevLayer.get_x()[np.newaxis]
		if self.currLayer.hasBias:
			x = np.append(x, [1])

#		print "x = ", x
#		z = np.dot(x, self.w.T)
		z = np.dot(self.w.T, x)
#		print "z = ", z
		
		# compute and store output
		y = self.act.func(z)
		#print "y = ", y
		self.currLayer.set_x(y)

		# compute and store output derivatives
		self.currLayer.set_deriv(self.act.deriv(z))

		return y
	
	def bprop(self, ni, target = None, verbose = False):
		yj = self.currLayer.get_x()
		if verbose: 
			print "out = ", yj
			print "w = ", self.w
		# compute error
		if self.currLayer.isOutput:
			if target is None: raise Exception("bprop(): target values needed for output layer")
			err = -(target - yj)
		else:
			err = self.currLayer.get_dErr()

		# compute gradients out

		if verbose: print "dE_dyj = ", dE_dyj

		# compute dE_dzj
		yj = self.currLayer.get_x()
		dE_dzj = yj * (1 - yj) * dE_dyj
		if verbose: print "dE_dzj = ", dE_dzj

		# backpropagate to dE/dyi
		yi = self.prevLayer.get_x()
		yi = np.append(yi, [1])
		if verbose: print "yi = ", yi

		if len(yi) != self.w.shape[0]: raise Exception("bprop(): layer sizes mismatch")
		dE_dyi = np.zeros(self.w.shape[0])

		for i in range(self.w.shape[0]):
			dE_dyi[i] += sum(self.w[i] * dE_dzj)

		if verbose: print "dE_dyi = ", dE_dyi
		self.prevLayer.set_dErr(np.delete(dE_dyi,-1))

		# computing dE/dw
		dE_dw = np.zeros(self.w.shape)
		for i in range(self.w.shape[0]):
			dE_dw[i] = yi[i] * dE_dyj

		dw = ni * dE_dw
		
		if verbose: print "ni * dE_dw = ", dw
		self.w -= dw

if __name__ == "__main__":		

	inputs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
	outs   = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

	layer0 = layer1D(2, isInput = True, x = inputs[0])
	layer1 = layer1D(4)
	layer2 = layer1D(100)
	layer3 = layer1D(2, isOutput = True )

	subnet01 = fullConnection(layer0, layer1)
	#subnet01 = fullConnection(layer0, layer1, w = np.array([[0, 100, -100, 0], [0, -100, 100, 0], [0, -5, -5, -5]]) )

	subnet12 = fullConnection(layer1, layer2) 
	#subnet12 = fullConnection(layer1, layer2, w = np.array([[1],[10], [10], [1], [-5]])  )

	subnet23 = fullConnection(layer2, layer3) 

	ni = 0.1
	for i in range(100):
		#sample = np.random.randint(len(inputs))
		sample = i % 4

		layer0.set_x(inputs[sample])

		#print "\nIN -> HIDDEN"
		subnet01.propagate()
		#print "\nHIDDEN -> OUT"
		subnet12.propagate()
		subnet23.propagate()
		#print "\n---------------"
#		print o, "\t",
#		if o > o_prev:
#			print "+"
#		elif o < o_prev:
#			print "-"
#		else:
#			print ""
#		o_prev = o
	
		#print "||BPROP 1"
		subnet23.bprop(ni, outs[sample], verbose = False)
	
		#print "\n||BPROP 2"
		subnet12.bprop(ni)
		subnet01.bprop(ni)
	
		#print ""

	for i in range(4):
		layer0.set_x(inputs[i])
		subnet01.propagate()
		subnet12.propagate()
		o = subnet23.propagate()
		print "In = ", inputs[i], "target = ", np.argmax(outs[i]), " predicted = ", np.argmax(o), "\t details = ", o
	


	
