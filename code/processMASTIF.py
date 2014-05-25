#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv

mastifDir = "../data/mastif_raw/TS2010/SourceImages"

minW = 32
minH = 32
rotDev = 0.08726
locDev = 1

# extracts part of the image defined by a (rotated) rectangualar
def extract(img, x, y, theta, w, h):
	center = (x + w/2.0, y + h/2.0)
	out = cv.CreateImage((w, h), img.depth, img.nChannels)
	mapping = np.array([[np.cos(theta), -np.sin(theta), center[0]], [np.sin(theta), np.cos(theta), center[1]]])
	map_matrix = cv.fromarray(mapping)
	cv.GetQuadrangleSubPix(img, out, map_matrix)
	return out

allClasses = []

datasetImages = []
datasetLabels = []

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
					if w >= minW and h >= minH:
						x = int(int(signMatch.group(2)) + np.random.normal(0, locDev))
						y = int(int(signMatch.group(3)) + np.random.normal(0, locDev))
						w = int(int(signMatch.group(4)) + np.random.normal(0, locDev))
						h = int(int(signMatch.group(5)) + np.random.normal(0, locDev))
	
						theta = np.random.normal(0, rotDev)
	
						print fName, ":", label, "at", x,",", y, "size", w, ",", h
						datasetLabels.append(label)
	
						# generate a unique set of existing labels
						if not (label in allClasses):
							allClasses.append(label)
	
						# load the image
						im = cv.LoadImage(mastifDir+"/"+fName)
						patch = extract(im, x, y, theta, w, h)
						cv.SaveImage("tmp/"+fName+".png", patch)
			
			else:
				raise Exception("Error processing MASTIF TS2010, line:"+line)

# compute dataset statistic
print "Total of", len(allClasses), "classes"
hist = np.zeros(len(allClasses))
for i in range(len(datasetLabels)):
	hist[allClasses.index(datasetLabels[i])] += 1

print "min = ", min(hist)
print "max = ", max(hist)
histFiltered = hist[hist > 100]
print "#classes > 100 = ", len(histFiltered), "= {"
for i in range(len(hist)):
	if hist[i] > 100: print allClasses[i], "(", hist[i], ")"
print "}"

width = 10
fig = plt.figure(num=None, figsize=(50, 6), dpi=80, facecolor='w', edgecolor='k')
x = np.arange(1, len(allClasses)+1)*10
bar1 = plt.bar(x, hist, width, color="y" )
plt.ylabel( '# occurrences' )
plt.xticks(x + width/2.0, allClasses )
plt.savefig("hist_MASTIF.png")
