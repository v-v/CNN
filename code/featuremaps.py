#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from utils import *

class layerFM:
	def __init__(self, n, width, height, isInput = False, isOutput = False):
	#	if n != 1 and isInput: raise Exception("Input layer can have only one feature map")
		self.n = n
		self.width = width
		self.height = height
		self.FMs = np.zeros([n, width, height])
		self.error = np.zeros([n, width, height])
		self.isInput = isInput
		self.isOutput = isOutput
	
	def shape(self):
		return [self.height, self.width]	
	
	def get_n(self):
		return self.n

	def resetError(self):
		self.error = np.zeros([self.n, self.width, self.height])
	
	def addError(self, error):
		self.error += error
	
	def get_FM_error(self):
		return self.error
	
	def set_FM_error(self, error):
		self.error = error
	
	def set_x(self, x):
		if x.shape != self.FMs.shape: raise Exception("FeatureMap: set_x dimensions do not match")
		self.FMs = x
	
	def get_FM(self):
		return self.FMs


	# compatibility functions for connecting to 1D layers
	def get_x(self):
		x = np.squeeze(self.FMs)
		if x.ndim != 1: raise Exception("Only 1x1 feature maps can be passed to a fully connected (1D) layer")
		return x
	
	def get_size(self):
		if self.height != 1 or self.width != 1: raise Exception("Only 1x1 feature maps can be connected to a fully connected (1D) layer")
		return self.n
	
	def set_error(self, err):
		self.error = err.reshape([len(err),1, 1,])
