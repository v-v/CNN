#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import os
import sys
import numpy as np
from cv2 import cv
from cv2 import split as cv2split



# --------------------------------
#          configuration
# --------------------------------

# output file
# stores all the information in an unified forma
outFile = '../data/labels.txt'


# creates images in per-class directories
# for dataset visual inspection
createImages = False
imagesRootDir = '../data/MASTIFImagesDump'
datasetDir = '../data/mastif_raw/TS2010/SourceImages'

# dataset descriptions (by Josip Krepac)
boundingBoxConfigFile = '../remapping/bb_config.txt'
imagesRemapFile        = '../remapping/image_remap.txt'
signsRemapFile        = '../remapping/signs_remap.txt'
setIndicatorFile      = '../remapping/config.txt'



# extracts part of the image defined by a (rotated) rectangualar
def extract(img, x, y, theta, w, h):
	center = (x + w/2.0, y + h/2.0)
	out = cv.CreateImage((w, h), img.depth, img.nChannels)
	mapping = np.array([[np.cos(theta), -np.sin(theta), center[0]], [np.sin(theta), np.cos(theta), center[1]]])
	map_matrix = cv.fromarray(mapping)
	cv.GetQuadrangleSubPix(img, out, map_matrix)
	return out


# --------------------------------
#        actual conversion
# --------------------------------


# read classes remapping
signsMap = {}
with open(signsRemapFile) as f:
	for line in f:
		mapMatch = re.search('(.*)\t(.*)', line)
		if mapMatch:
		    signOld = mapMatch.group(1)
		    signNew = mapMatch.group(2)

		    signsMap[signOld] = signNew

# read images remapping
imagesMap = {}
with open(imagesRemapFile) as f:
	for line in f:
		mapMatch = re.search('(.*) (.*)', line)
		if mapMatch:
		    nameOld = mapMatch.group(1)
		    nameNew = mapMatch.group(2)

		    imagesMap[nameOld] = nameNew + '.bmp'


# read the dataset division
datasetMap = {}
with open(setIndicatorFile) as f:
	for line in f:
		mapMatch = re.search('(.*)\t(.*)\t(.*)', line)
		if mapMatch:
			name = mapMatch.group(1)
			dataset = mapMatch.group(2)
			if dataset == '0':
				datasetMap[name] = "Test"
			elif dataset == '1':
				datasetMap[name] = "Train"
			else:
				print "Wrong dataset indicator:", dataset
				sys.exit(-1)

# read the signs and generate the output set description
fOut = open(outFile, "w")

if createImages:
	import shutil
	try: 
		shutil.rmtree(imagesRootDir)
	except:
		pass
	os.mkdir(imagesRootDir)

with open(boundingBoxConfigFile) as f:
	for line in f:
		match = re.search('(.*)\t(.*)\t(\d+),(\d+),(\d+),(\d+)', line)
		if match:
			imageName =  match.group(1)
			sampleClass = match.group(2)
			sampleX = match.group(3)
			sampleY = match.group(4)
			sampleW = match.group(5)
			sampleH = match.group(6)

			print imagesMap[imageName], signsMap[sampleClass], datasetMap[imageName],  "\tx=", sampleX, "y=",sampleY, "w=",sampleW, "h=",sampleH
			fOut.write(imagesMap[imageName] + "\t" +     \
			            signsMap[sampleClass] + "\t" +   \
			            datasetMap[imageName] + "\t" +   \
				    sampleX + "\t" + sampleY + "\t" + \
				    sampleW + "\t" + sampleH + "\n")

			if createImages:
				im = cv.LoadImage(datasetDir+"/"+imagesMap[imageName])
				sample = extract(im, int(sampleX), int(sampleY), 0, int(sampleW), int(sampleH))

				if not os.path.isdir(imagesRootDir+"/"+signsMap[sampleClass]):
					os.mkdir(imagesRootDir+"/"+signsMap[sampleClass])
				cv.SaveImage(imagesRootDir+"/"+signsMap[sampleClass]+"/"+imageName+".png",sample)
				



fOut.close()
