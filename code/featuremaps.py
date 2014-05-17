#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from utils import *

class layerFM:
	def __init__(self, n, width, height, isInput = False):
		if n != 1 and isInput: raise Exception("Input layer can have only one feature map")
		self.n = n
		self.width = width
		self.height = height
		self.FMs = np.zeros([n, width, height])
		self.error = np.zeros([n, width, height])
	
	def get_n(self):
		return self.n

	def resetError(self):
		self.error = np.zeros([self.n, self.width, self.height])
	
	def addError(self, error):
		self.error += error
	
	def get_error(self):
		return get_error
	
	def set_error(self, error):
		self.error = error
	
	def set_x(self, x):
		if x.shape != self.FM.shape: raise Exception("FeatureMap: set_x dimensions do not match")
		self.FMs = x
	
	def get_x(self):
		return self.FMs
