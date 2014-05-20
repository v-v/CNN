#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from utils import *

class convolutionalConnection:
	def __init__(self, prevLayer, currLayer, poolingWindowShape):
		self.prevLayer   = prevLayer
		self.currLayer  = currLayer
		self.poolingWindowShape = poolingWindowShape
	
	def propagate(self):
		[prevSizeX, prevSizeY] = self.prevLayer.shape()
		
		return None
	
	def bprop(self):
		return None


