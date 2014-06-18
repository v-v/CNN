#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
from cv2 import cv
from cv2 import split as cv2split

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import cPickle
import gzip

# --------------------------------
#          configuration
# --------------------------------
mastifDir = "../data/mastif_raw/TS2010/SourceImages"

trainSetFName = "../data/mastif_ts2010_train.pkl"
testSetFName  = "../data/mastif_ts2010_test.pkl"
evalSetFName  = "../data/mastif_ts2010_eval.pkl"

# a class should have
limit = 30 	# at least this many images
reqImgSize = 44 # of at laste this size (in both sets)

padding = 2

rotDev = 0.08726
locDev = 1

# generate additional samples by 
# distorting existing ones until
# the requrired number of samples
# per class is met

spcTrain   = 500
spcTest    = 100
# --------------------------------


# storage for sample parameters
class image:
	def __init__(self, fName, x, y, w, h, theta):
		self.fName = fName
		self.x = x
		self.y = y
		self.w = w
		self.h = h
		self.theta = theta

# extracts part of the image defined by a (rotated) rectangualar
def extract(img, x, y, theta, w, h):
	center = (x + w/2.0, y + h/2.0)
	out = cv.CreateImage((w, h), img.depth, img.nChannels)
	mapping = np.array([[np.cos(theta), -np.sin(theta), center[0]], [np.sin(theta), np.cos(theta), center[1]]])
	map_matrix = cv.fromarray(mapping)
	cv.GetQuadrangleSubPix(img, out, map_matrix)
	return out


allClasses = []
trainSetStats = []
testSetStats = []

trainSet = defaultdict(list)
testSet = defaultdict(list)

with open(mastifDir+'/index.txt') as f:
	for line in f:
		match = re.search('(.*)\t(.*)\t(.*)\t(\d+)\t(\d+)\t(\d+)\t(\d+)', line)
		if match:
			fName = match.group(1)
			label = match.group(2).upper()
			dataset = match.group(3).lower()
			posX  = int(match.group(4))
			posY  = int(match.group(5))
			sizeX  = int(match.group(6))
			sizeY  = int(match.group(7))

			if sizeX >= reqImgSize and sizeY >= reqImgSize:
				# generate a unique set of existing labels
				if not (label in allClasses):
					allClasses.append(label)
	
				# for statistic purposes
				if dataset == 'train':
					trainSetStats.append(label)
				elif dataset == 'test':
					testSetStats.append(label)
				else:
					print "Unknown dataset", dataset
					sys.exit(-1)
	

# generate statistics
print "The dataset contains a total of", len(allClasses), "classes where images are bigger or equal than", reqImgSize
histTrain = np.zeros(len(allClasses))
histTest = np.zeros(len(allClasses))
for i in range(len(trainSetStats)):
	histTrain[allClasses.index(trainSetStats[i])] += 1

for i in range(len(testSetStats)):
	histTest[allClasses.index(testSetStats[i])] += 1

# training set histogram
width = 10
fig = plt.figure(num=None, figsize=(40, 6), dpi=80, facecolor='w', edgecolor='k')
x = np.arange(1, len(allClasses)+1)*10
bar1 = plt.bar(x, histTrain, width, color="y" )
plt.ylabel( 'broj pojavljivanja' )
plt.ylabel( 'oznaka znaka (skup za ucenje)' )
plt.xticks(x + width/2.0, allClasses )
plt.savefig("hist_MASTIF_train.png")

# test set histogram
fig = plt.figure(num=None, figsize=(40, 6), dpi=80, facecolor='w', edgecolor='k')
x = np.arange(1, len(allClasses)+1)*10
bar1 = plt.bar(x, histTest, width, color="y" )
plt.ylabel( 'broj pojavljivanja' )
plt.ylabel( 'oznaka znaka (skup za ispitivanje)' )
plt.xticks(x + width/2.0, allClasses )
plt.savefig("hist_MASTIF_test.png")

acceptedClasses = []
print "\nClasses with more than", str(limit), "samples of size >=",reqImgSize,"= {"
for i in range(len(allClasses)):
	if histTrain[i] > limit and histTest[i]:
		print allClasses[i], "(Train =", int(histTrain[i]),"Test =", int(histTest[i]), ")"
		acceptedClasses.append(allClasses[i])
print "}"


# agregate samples within the accepted classes
trainSet = defaultdict(list)
testSet = defaultdict(list)

print "\nAgreggating suitable samples...",
with open(mastifDir+'/index.txt') as f:
	for line in f:
		match = re.search('(.*)\t(.*)\t(.*)\t(\d+)\t(\d+)\t(\d+)\t(\d+)', line)
		if match:
			fName = match.group(1)
			label = match.group(2).upper()
			dataset = match.group(3).lower()
			x  = int(match.group(4))
			y  = int(match.group(5))
			w  = int(match.group(6))
			h  = int(match.group(7))

			if (w >= reqImgSize) and (h >= reqImgSize) and (label in acceptedClasses):
				sample = image(fName, x, y, w, h, 0)
				if dataset == 'train':
					trainSet[label].append(sample)
				elif dataset == 'test':
					testSet[label].append(sample)
				else:
					print "Unknown dataset (this shuldn'g have happened!)", dataset
					sys.exit(-1)
print "done"

# expand dataset by adding distorted existing samples
# ---------------------------------------------------

def expandDataset(dataset, spc):
	for c in dataset:
		curr = len(dataset[c])
		print c, "constains", curr, "elements, generating "+str(spc - curr)+" additional samples"
		for i in range(spc - curr):
			rnd = np.random.randint(0, curr)
			fName = dataset[c][rnd].fName
			x = int(dataset[c][rnd].x + np.random.normal(0, locDev))
			y = int(dataset[c][rnd].y + np.random.normal(0, locDev))
			w = int(dataset[c][rnd].w + np.random.normal(0, locDev))
			h = int(dataset[c][rnd].h + np.random.normal(0, locDev))
			theta = np.random.normal(0, rotDev)
			sample = image(fName, x, y, w, h, theta)

			dataset[c].append(sample)

print "\nAnalyzing training set:"
expandDataset(trainSet, spcTrain)

print "\nAnalyzing test set:"
expandDataset(testSet, spcTest)

# generate actual samples
# -----------------------

def normalize3(images):
	norm = 1.0 * images[0][0].shape[0] * images[0][0].shape[1]
	for img in range(len(images)):
		for ch in range(len(images[img])):
			s = np.sum(images[img][ch])
			m = s / norm
	
			# computing std
			std = 0.0
	
			for i in range(images[0][0].shape[0]):
				for j in range(images[0][0].shape[1]):
					std += (images[img][ch][i][j] - m) ** 2.0
	
			std = np.sqrt(std/norm)
	
			# mean and std normalization
			for i in range(images[0][0].shape[0]):
				for j in range(images[0][0].shape[1]):
					images[img][ch][i][j] = (images[img][ch][i][j] - m)/std 
	

def generateData(dataset):
	data = []
	labels = []

	fig = plt.figure(num=None, figsize=(5, 10), dpi=80, facecolor='w', edgecolor='k')

	for c in dataset:
		print c, 
		sys.stdout.flush()		
		for i in range(len(dataset[c])):
			s = dataset[c][i]
			im = cv.LoadImage(mastifDir+"/"+s.fName)

			if s.w > s.h:
				offsetH = int((s.w - s.h)/2)
				patch = extract(im, s.x, s.y - offsetH, s.theta, s.w, s.w)
				currSize = s.w
			else:
				offsetW = int((s.h - s.w)/2)
				patch = extract(im, s.x+offsetW, s.y, s.theta, s.h, s.h)
				currSize = s.h

			scale = reqImgSize * 1.0 / currSize
			destSize = int(currSize * scale)

			patchResized = cv.CreateImage((destSize, destSize), patch.depth, patch.nChannels)
			cv.Resize(patch, patchResized,interpolation=cv.CV_INTER_LINEAR)


			x = np.asarray(patchResized[:,:])
			b,g,r = cv2split(x)

			imgB = np.zeros([reqImgSize + 2*padding, reqImgSize + 2*padding])
			imgB[padding:(padding + b.shape[0]), padding:(padding + b.shape[1])] = b

			imgG = np.zeros([reqImgSize + 2*padding, reqImgSize + 2*padding])
			imgG[padding:(padding + g.shape[0]), padding:(padding + g.shape[1])] = g


			imgR = np.zeros([reqImgSize + 2*padding, reqImgSize + 2*padding])
			imgR[padding:(padding + r.shape[0]), padding:(padding + r.shape[1])] = r

			data.append([imgB, imgG, imgR])

			label = np.zeros(len(dataset))
			label[acceptedClasses.index(c)] = 1

			labels.append(label)

	print "OK, normalizing...", 
	sys.stdout.flush()
	normalize3(data)

	return data, labels

print "\nGenerating datasets... "
print "Training set:",
sys.stdout.flush()
trainSetData, trainSetLabels = generateData(trainSet)
print "[done]\nTesting set:",
sys.stdout.flush()
testSetData, testSetLabels = generateData(testSet)
print "[done]\n",
sys.stdout.flush()

# saving
# ------
print "Saving training set to", trainSetFName, "..." 
sys.stdout.flush()
f = gzip.open(trainSetFName, 'wb')
cPickle.dump((trainSetData, trainSetLabels), f)

print "Saving testing set to", testSetFName, "..." 
sys.stdout.flush()
f = gzip.open(testSetFName, 'wb')
cPickle.dump((testSetData, testSetLabels), f)

# generating a few random samples from the dataset
# ------------------------------------------------
def displayRandomSamples(data, labels, fName):
	fig = plt.figure(num=None, figsize=(10, 40), dpi=80, facecolor='w', edgecolor='k')

	n = 12	
	for i in range(n):
		c = np.random.randint(len(data)-1)

		plt.subplot(n, 3, i*3 +1)
		plt.imshow(data[c][0], cmap=plt.cm.Blues)
		plt.title("i("+str(c)+", b)="+acceptedClasses[np.argmax(labels[c])])

		plt.subplot(n, 3, i*3 +2)
		plt.imshow(data[c][1], cmap=plt.cm.Greens)
		plt.title("i("+str(c)+", g)="+acceptedClasses[np.argmax(labels[c])])

		plt.subplot(n, 3, i*3 +3)
		plt.imshow(data[c][2], cmap=plt.cm.Reds)
		plt.title("i("+str(c)+", r)="+acceptedClasses[np.argmax(labels[c])])

	plt.savefig(fName)

print "\nGenerating a few sample images...", 
sys.stdout.flush()
displayRandomSamples(trainSetData, trainSetLabels, "trainSet_samples.png")
displayRandomSamples(testSetData, testSetLabels, "testSet_samples.png")
print "[done]"
