#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib as mlp
mlp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cPickle
import struct
import gzip


class data:

	def __init__(self):
		self.nRows = None
		self.nCols = None
		self.padded = None

	# loads a set in IDX format (used in MNIST) and saves it as a python packed list
	def convertIDX(self, fnameOut, fnameData, fnameLabels = None, normalize = False, padding = None):
		self.padded = padding
		f = open(fnameData, "rb")
		try:
			magic   = struct.unpack('!I', f.read(4))[0]
			if magic != 2051: raise Exception("File "+fnameData+"seems not to be an IDX picture file")
	
			nImages = struct.unpack('!I', f.read(4))[0]
			self.nRows   = struct.unpack('!I', f.read(4))[0]
			self.nCols   = struct.unpack('!I', f.read(4))[0]

			if padding is not None:
				print "Loading", nImages, "images and padding to size", self.nRows + padding, "x", self.nCols + padding
			else:
				print "Loading", nImages, "images of size", self.nRows, "x", self.nCols
	
			expectedBytes = nImages * self.nRows * self.nCols
			actualBytes = 0
	
			imgCount = 0
			pixelCount = 0
	
			# allocate image
			if padding is None:
				img = np.empty([self.nRows, self.nCols])
			else:
				img = np.zeros([self.nRows+padding*2, self.nCols+padding*2])

			images = []
			pixel = f.read(1)
	
			while pixel != "":
				pixel = struct.unpack('B', pixel)[0] #/ 255.0
				if pixelCount == self.nRows * self.nCols:
					pixelCount = 0
					imgCount += 1
					images.append(img)
					if padding is None:
						img = np.empty([self.nRows, self.nCols])
					else:
						img = np.zeros([self.nRows+padding*2, self.nCols+padding*2])
	
				i = np.floor(pixelCount / self.nRows).astype(int)
				j = pixelCount % self.nCols
	
				if padding is not None:
					img[i+padding, j+padding] = pixel
				else:
					img[i, j] = pixel
	
				pixel =  f.read(1)
				actualBytes += 1
				pixelCount += 1
	
		finally:
			f.close()
	
		# add last picture
		imgCount += 1
		images.append(img)
	
		if expectedBytes != actualBytes: raise Exception("Number of expected bytes differs from actual while loading "+ fnameData)
		if normalize:
			self.normalize(images)
#			print "Afer normalization:"
#			print images

		# process labels if available
		if fnameLabels != None:
			print "Loading labels"
			f = open(fnameLabels, "rb")
			try:
				magic   = struct.unpack('!I', f.read(4))[0]
				if magic != 2049: raise Exception("File "+fnameLabels+"seems not to be an IDX label file")
		
				nLabels = struct.unpack('!I', f.read(4))[0]
		
				expectedBytes = nLabels
				actualBytes = 0
		
				labels = []
				label = f.read(1)
		
				while label != "":
					label = struct.unpack('B', label)[0]
					labels.append(label)
						
					label =  f.read(1)
					actualBytes += 1
			finally:
				f.close()
	
			if expectedBytes != actualBytes: raise Exception("Number of expected bytes differs from actual while loading "+ fnameLabels)

			# multiple classes representation
			# 1 - belongs to class
			# 0 - all others
			labelsMultipleClasses = []
			nClasses = max(labels) - min(labels) + 1
	
			for i in range(nLabels):
				classes = np.zeros(nClasses)
				classes[labels[i]] = 1
				labelsMultipleClasses.append(classes)

			if len(labelsMultipleClasses) != len(images): raise Exception("Number of images and labels does not match")
	
		print "Saving as packaged list"	
	
		f = gzip.open(fnameOut, 'wb')
		if fnameLabels != None:
			cPickle.dump((images, labelsMultipleClasses), f)
		else:
			cPickle.dump((images), f)


	def normalize(self, images):
		norm = 1.0 * images[0].shape[0] * images[0].shape[1]
		for img in range(len(images)):
			s = np.sum(images[img])
			m = s / norm

			# computing std
			std = 0.0

			for i in range(self.nRows):
				for j in range(self.nCols):
					std += (images[img][i][j] - m) ** 2.0

			std = np.sqrt(std/norm)

			# mean and std normalization
			for i in range(images[0].shape[0]):
				for j in range(images[0].shape[1]):
					images[img][i][j] = (images[img][i][j] - m)/std 

	def loadData(self, fName):
		f = gzip.open(fName)
		data = cPickle.load(f)
		f.close()
		return data
	
	def filterClasses(self, images, labels, classesToKeep):
		mask = np.zeros([len(classesToKeep), labels[0].shape[0]])
		for i in range(len(classesToKeep)):
			mask[i][classesToKeep[i]] = 1

		imagesFiltered = []
		labelsFiltered = []
		for i in range(len(images)):
			for c in range(mask.shape[0]):
#				print "labels[i] =", labels[i], "mask =", mask
				if np.all(labels[i] == mask[c]):
					imagesFiltered.append(images[i])
					label = np.zeros(len(classesToKeep))
					label[c] = 1
					labelsFiltered.append(label)

		print "Filtered", len(images), "images to", len(imagesFiltered)
		return imagesFiltered, labelsFiltered
	
	def saveData(self, fName, images, labels):
		f = gzip.open(fName, 'wb')
		cPickle.dump((images, labels), f)

	# shows a few (n) samples from the given dataset (in pkl format)
	def visualizeRandomSamples(self, images, labels, n, fName = None):
		if len(images) != len(labels): raise Exception("visualizeRandomSamples: number of images and labels doesn't match")
		from pprint import pprint
		for i in range(n):
			c = np.random.randint(len(images)-1)
			plt.subplot(1, n, i+1)
			plt.imshow(images[c], cmap=plt.cm.gray)
			pprint(images[c])
			plt.title("image("+str(c)+")="+str(np.argmax(labels[c])))

		if fName:
			plt.savefig(fName)
		else:
			plt.show()

		
		
if __name__ == "__main__":		
	
	d = data()
	
	#d.convertIDX("../data/MNISTtrain-norm.pkl", "../data/train-images.idx3-ubyte", "../data/train-labels.idx1-ubyte", normalize = True, padding = 2)
	#d.convertIDX("../data/MNISTtest-norm.pkl", "../data/t10k-images.idx3-ubyte", "../data/t10k-labels.idx1-ubyte", normalize = True, padding = 2)
	
#	images, labels = d.loadData("../data/MNISTtrain-norm.pkl")
#	images, labels = d.filterClasses(images, labels, [0, 1])
#	d.saveData("../data/MNISTtrain-2-classes-norm.pkl", images, labels)

	images, labels = d.loadData("../data/MNISTtrain-2-classes-norm.pkl")

	print "images = ", len(images), images[0].shape
	print "labels = ", len(labels), labels[0].shape

	d.visualizeRandomSamples(images, labels, 3 )
	#d.visualizeRandomSamples(images, labels, 3, "out.png" )

