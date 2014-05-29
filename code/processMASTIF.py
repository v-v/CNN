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

reqImgSize = 44
padding = 2

rotDev = 0.08726
locDev = 1

# generate additional samples by 
# distorting existing ones until
# the requrired number of samples
# per class is met

spcTrain   = 500
spcTest    = 100
spcEval    = 100

percentTrain = 70
percentTest = 15
percentEval = 15

# classes that we want to include in the set
#acceptedClasses = ['B32', 'B31', 'A04', 'C44', 'C79', 'C80', 'C86', 'C02', 'A33', 'C11', 'A05', 'B46', 'D10', 'A03', 'E03' ]
acceptedClasses = ['A33', 'B31', 'A05', 'B28', 'B32', 'B46', 'A03', 'A44', 'A04', 'C11', 'C02', 'A11']

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


# Aggregate samples
# -----------------
allClasses = []
datasetImages = []
datasetLabels = []

trainSet = defaultdict(list)
testSet = defaultdict(list)
evalSet = defaultdict(list)


with open(mastifDir+'/index.seq') as f:
    for line in f:
	    imageMatch = re.search('\[(.*)\]\:(.*)', line)
	    if imageMatch:
		    fName = imageMatch.group(1)
		    tmp = imageMatch.group(2)
		    signs = tmp.split("&")
		    for sign in signs:
			signMatch = re.search('(.*)@\(x=(.*),y=(.*),w=(.*),h=(.*)\)', sign)
			if signMatch:
				label = signMatch.group(1)
				if label != "": # why do some entries lack a label?
					x =int(signMatch.group(2))
					y =int(signMatch.group(3))
					w =int(signMatch.group(4))
					h =int(signMatch.group(5))

					if w >= reqImgSize and h >= reqImgSize:
						# generate a unique set of existing labels
						if not (label in allClasses):
							allClasses.append(label)
						
						datasetLabels.append(label)

						if label in acceptedClasses:
							
							#print fName, ":", label, "at", x,",", y, "size", w, ",", h
	
							sample = image(fName, x, y, w, h, 0)
							rnd = np.random.uniform(0, percentTrain + percentTest + percentEval)
							if rnd >= 0 and rnd <= percentTrain:
								trainSet[label].append(sample)
							elif rnd > percentTrain and rnd <= percentTrain + percentTest:
								testSet[label].append(sample)
							else:
								evalSet[label].append(sample)
			else:
				raise Exception("Error processing MASTIF TS2010, line:"+line)

# compute dataset statistic
# -------------------------
print "The dataset contains a total of", len(allClasses), "classes for images bigger than", str(reqImgSize), "x", str(reqImgSize)
hist = np.zeros(len(allClasses))
for i in range(len(datasetLabels)):
	hist[allClasses.index(datasetLabels[i])] += 1

print "\nLeast samples per class = ", int(min(hist))
print "\nMaximum samples per class = ", int(max(hist))
limit = 35
histFiltered = hist[hist > limit]
print "\nClasses with more than", str(limit), "samples (", len(histFiltered), ") = {"
for i in range(len(hist)):
	if hist[i] > limit: print allClasses[i], "(", int(hist[i]), ")"
print "}"

width = 10
fig = plt.figure(num=None, figsize=(50, 6), dpi=80, facecolor='w', edgecolor='k')
x = np.arange(1, len(allClasses)+1)*10
bar1 = plt.bar(x, hist, width, color="y" )
plt.ylabel( '# occurrences' )
plt.xticks(x + width/2.0, allClasses )
plt.savefig("hist_MASTIF.png")

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

print "\nAnalyzing evaluation set:"
expandDataset(evalSet, spcEval)

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
trainSetData, trainSetLabels = generateData(trainSet)
print "[done]\nTesting set:",
testSetData, testSetLabels = generateData(testSet)
print "[done]\nEvaluation set:",
evalSetData, evalSetLabels = generateData(evalSet)
print "[done]\n"
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


print "Saving evaluation set to", evalSetFName, "..." 
sys.stdout.flush()
f = gzip.open(evalSetFName, 'wb')
cPickle.dump((evalSetData, evalSetLabels), f)




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
displayRandomSamples(evalSetData, evalSetLabels, "evalSet_samples.png")
print "[done]"
