#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Convolutional Neural Network library
# copyright (c) 2014 Vedran Vukotic
# gmail: vevukotic

# pooling.py - implements forward and backward pooling operations

import numpy as np
from utils import *

class poolingConnection:
	def __init__(self, prevLayer, currLayer, poolingStepX, poolingStepY):
		self.prevLayer   = prevLayer
		self.currLayer  = currLayer
		self.poolingStepX = poolingStepX
		self.poolingStepY = poolingStepY

		if self.prevLayer.shape()[0] / poolingStepY != self.currLayer.shape()[0] or \
		   self.prevLayer.shape()[1] / poolingStepX != self.currLayer.shape()[1]:
			raise Exception('Pooling step should match size ratio between consecutive layers')

		if self.prevLayer.get_n() != self.currLayer.get_n():
			raise Exception('Number of feature maps before and after pooling should be the same')
	
	def propagate(self):
		[prevSizeY, prevSizeX] = self.prevLayer.shape()
		[currSizeY, currSizeX] = self.currLayer.shape()

		self.maximaLocationsX = np.zeros([self.currLayer.get_n(), self.currLayer.shape()[0], self.currLayer.shape()[1]])
		self.maximaLocationsY = np.zeros([self.currLayer.get_n(), self.currLayer.shape()[0], self.currLayer.shape()[1]])

		pooledFM = np.zeros([self.currLayer.get_n(), self.currLayer.shape()[0], self.currLayer.shape()[1]])

		yi = self.prevLayer.get_FM()

		for n in range(self.prevLayer.get_n()):
			for i in range(currSizeY):
				for j in range(currSizeX):
					reg = yi[n, i*self.poolingStepY:(i+1)*self.poolingStepY, j*self.poolingStepX:(j+1)*self.poolingStepX]
					loc = np.unravel_index(reg.argmax(), reg.shape) + np.array([i*self.poolingStepY, j*self.poolingStepY])
					self.maximaLocationsY[n, i, j] = loc[0]
					self.maximaLocationsX[n, i, j] = loc[1]
					pooledFM[n, i, j] = yi[n, loc[0], loc[1]]
	
		self.currLayer.set_FM(pooledFM)

	def bprop(self):
		currErr = self.currLayer.get_FM_error()
#		print "Error in front = \n", currErr
		prevErr = np.zeros([self.prevLayer.get_n(), self.prevLayer.shape()[0], self.prevLayer.shape()[1]])
		
		[currSizeY, currSizeX] = self.currLayer.shape()

		for n in range(self.prevLayer.get_n()):
			for i in range(currSizeY):
				for j in range(currSizeX):
					prevErr[n, self.maximaLocationsY[n, i, j], self.maximaLocationsX[n, i, j]] = currErr[n, i, j]
#					prevErr[n, self.maximaLocationsY[n, i, j], self.maximaLocationsX[n, i, j]] = currErr[n, j, i]

		self.prevLayer.set_FM_error(prevErr)
#		print "Error back = \n", prevErr


