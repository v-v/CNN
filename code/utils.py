#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Convolutional Neural Network library
# copyright (c) 2014 Vedran Vukotic
# gmail: vevukotic

# utils.py - defines activation fuctions, their derivatives,
# random initlialization ranges and other useful things

import numpy as np

class logistic:
	# initialize a logistic activation fuction
	# with beta as a slope parameter
	def __init__(self, beta = 1.0):
		self.beta = beta

	def func(self, x):
		return 2.0 / (1.0 + np.exp(-self.beta * x)) - 1.0
	
	def deriv(self, x):
		y = self.func(x)
		e = np.exp(-self.beta * x)
		return 2.0 * self.beta * x * e / ((1+e) ** 2.0)
	
	# defines uniform sampling intervals for weights
	# initialization based on the number of neurons
	def sampleInterval(self, prev, curr):
		d = (- 4.0) * np.sqrt(6.0 / (prev + curr))
		return [-d, d]

class tanh:
	def func(self, x):
		return 1.7159 * np.tanh(2.0 * x / 3.0)
	
	def deriv(self, x):
		t = np.tanh(2.0 * x / 3.0) ** 2.0
		return 1.144 * (1 - t)

	def sampleInterval(self, prev, curr):
		d = (- 1.0) * np.sqrt(6.0 / (prev + curr))
		return [-d, d]
