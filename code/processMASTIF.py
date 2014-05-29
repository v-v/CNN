#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from cv2 import cv
from cv2 import split as cv2split

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# --------------------------------
#          configuration
# --------------------------------
mastifDir = "../data/mastif_raw/TS2010/SourceImages"

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
acceptedClasses = ['B32', 'B31', 'A04', 'C44', 'C79', 'C80', 'C86', 'C02', 'A33', 'C11', 'A05', 'B46', 'D10', 'A03', 'E03' ]

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

					# generate a unique set of existing labels
					if not (label in allClasses):
						allClasses.append(label)
					
					datasetLabels.append(label)

					if label in acceptedClasses:
						x =int(signMatch.group(2))
						y =int(signMatch.group(3))
						w =int(signMatch.group(4))
						h =int(signMatch.group(5))
						if w >= reqImgSize and h >= reqImgSize:
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
print "The dataset contains a total of", len(allClasses), "classes"
hist = np.zeros(len(allClasses))
for i in range(len(datasetLabels)):
	hist[allClasses.index(datasetLabels[i])] += 1

print "\nLeast samples per class = ", int(min(hist))
print "\nMaximum samples per class = ", int(max(hist))
histFiltered = hist[hist > 100]
print "\nClasses with more than 100 samples (", len(histFiltered), ") = {"
for i in range(len(hist)):
	if hist[i] > 100: print allClasses[i], "(", int(hist[i]), ")"
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

def generateData(dataset):
	data = []
	labels = []

	fig = plt.figure(num=None, figsize=(5, 10), dpi=80, facecolor='w', edgecolor='k')

	for c in dataset:
		print c, 
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

#			print b.shape, g.shape, r.shape


			plt.subplot(3, 1, 1)
			plt.axis('off')
			plt.imshow(imgB, cmap=plt.cm.gray)
			plt.subplot(3, 1, 2)
			plt.axis('off')
			plt.imshow(imgG, cmap=plt.cm.gray)
			plt.subplot(3, 1, 3)
			plt.axis('off')
			plt.imshow(imgR, cmap=plt.cm.gray)
			plt.savefig("tmp/"+c+"_"+str(i).zfill(3)+"_1.png")

			cv.SaveImage("tmp/"+c+"_"+str(i).zfill(3)+"_0.png", patch)
	print 
	return None, None

print "\nGenerating datasets... "
print "Training set",
trainSetData, trainSetLabels = generateData(trainSet)
print "done\nTest set",
testSetData, testSetLabels = generateData(testSet)
print "done\nEvaluation set",
evalSetData, evalSetLabels = generateData(evalSet)
print "done"
