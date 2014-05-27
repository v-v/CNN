#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from cv2 import cv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# --------------------------------
#          configuration
# --------------------------------
mastifDir = "../data/mastif_raw/TS2010/SourceImages"

minW = 32
minH = 32
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
				#		if w >= minW and h >= minH:
						if True:
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

print "\nProcessing training set:"
expandDataset(trainSet, spcTrain)

print "\nProcessing test set:"
expandDataset(testSet, spcTest)

print "\nProcessing evaluation set:"
expandDataset(evalSet, spcEval)

# generate actual samples
# -----------------------
