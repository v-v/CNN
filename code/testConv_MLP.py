#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from conv import *
from mlp import *

if __name__ == "__main__":

	in_data = np.array([
	      [
	      [[0, 0, 1, 1, 0, 0],
	       [0, 1, 1, 1, 0, 0],
	       [0, 0, 1, 1, 0, 0],
	       [0, 0, 1, 1, 0, 0],
	       [0, 0, 1, 1, 0, 0],
	       [0, 1, 1, 1, 1, 0]],

	      ],
	      
	      [
	      [[0, 0, 1, 1, 0, 0],
	       [0, 1, 1, 1, 1, 0],
	       [0, 1, 0, 1, 1, 0],
	       [0, 0, 1, 1, 0, 0],
	       [0, 1, 1, 0, 0, 0],
	       [0, 1, 1, 1, 1, 0]]

	      ]
	      ])

	out_data = np.array( [
	     [1, 0],

	     [0,1]
	])


	inLayer = layerFM(1, 6, 6, isInput = True)
	
	convLayer1 = layerFM(4, 3, 3)
	convLayer2 = layerFM(4, 1, 1)
	fullLayer1 = layer1D(10)
	fullLayer2 = layer1D(10)
	fullLayer3 = layer1D(2, isOutput = True )


	conv01 = convolutionalConnection(inLayer, convLayer1, np.ones([1, 4]), 4, 4, 1, 1)	
	conv12 = convolutionalConnection(convLayer1, convLayer2, np.ones([4, 4]), 3, 3, 1, 1)
	
	subnet01 = fullConnection(convLayer2, fullLayer1)
	subnet12 = fullConnection(fullLayer1, fullLayer2) 
	subnet23 = fullConnection(fullLayer2, fullLayer3) 


	ni = 0.005

	for it in range(500):
		#print "\nout = \n", conv1.propagate()
		inLayer.set_FM(in_data[0])
		conv01.propagate()
		conv12.propagate()
		subnet01.propagate()
		subnet12.propagate()
		subnet23.propagate()
		mse1 = fullLayer3.sampleMSE(out_data[0])

		subnet23.bprop(ni, out_data[0])
		subnet12.bprop(ni)
		subnet01.bprop(ni)
		conv12.bprop(ni)
		conv01.bprop(ni)
		
		inLayer.set_FM(in_data[1])
		conv01.propagate()
		conv12.propagate()
		subnet01.propagate()
		subnet12.propagate()
		subnet23.propagate()
		mse2 = fullLayer3.sampleMSE(out_data[1])

		subnet23.bprop(ni, out_data[1])
		subnet12.bprop(ni)
		subnet01.bprop(ni)
		conv12.bprop(ni)
		conv01.bprop(ni)

# ---------------------------
		if it % 10 == 0:

			print "it =", it, "MSE =", mse1, mse2
			 
			plt.subplot(6, 4, 1)
			plt.axis('off')
			plt.imshow(in_data[0][0], cmap=plt.cm.gray)
			
			plt.subplot(6, 4, 2)
			plt.axis('off')
			plt.imshow(in_data[1][0], cmap=plt.cm.gray)
		
			plt.subplot(6, 4, 3)
			plt.axis('off')
			plt.imshow(in_data[0][0], cmap=plt.cm.gray)
			
			plt.subplot(6, 4, 4)
			plt.axis('off')
			plt.imshow(in_data[1][0], cmap=plt.cm.gray)
		
			for i in range(conv01.k.shape[0]):
				plt.subplot(6, 4, i+5)
				plt.axis('off')
				#plt.imshow(conv1.k[i], cmap=plt.cm.gray, interpolation='none')
				plt.imshow(conv01.k[i], cmap=plt.cm.gray)
			
			for i in range(conv12.k.shape[0]):
				plt.subplot(6, 4, i+9)
				plt.axis('off')
				#plt.imshow(conv2.k[i], cmap=plt.cm.gray, interpolation='none')
				plt.imshow(conv12.k[i], cmap=plt.cm.gray)
			
			
		
			#plt.show()
			plt.savefig("imgs/"+str(it).zfill(6)+"kernels.png")


# ----------------------------
	#print "\nout= \n", conv1.propagate()
	
	np.set_printoptions(precision=3)

	inLayer.set_FM(in_data[0])
	conv01.propagate()
	conv12.propagate()
	subnet01.propagate()
	subnet12.propagate()
	print subnet23.propagate()
	
	inLayer.set_FM(in_data[1])
	conv01.propagate()
	conv12.propagate()
	subnet01.propagate()
	subnet12.propagate()
	print subnet23.propagate()
	

