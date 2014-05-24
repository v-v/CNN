#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import numpy as np
import matplotlib.pyplot as plt


fNameIn = "../data/mastif_raw/TS2010/SourceImages/index.seq"

allClasses = []

datasetImages = []
datasetLabels = []

with open(fNameIn) as f:
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
					x = signMatch.group(2)
					y = signMatch.group(3)
					w = signMatch.group(4)
					h = signMatch.group(5)

					print fName, ":", label, "at", x,",", y, "size", w, ",", h
					datasetLabels.append(label)

					# generate a unique set of existing labels
					if not (label in allClasses):
						allClasses.append(label)
			
			else:
				raise Exception("Error processing MASTIF TS2010, line:"+line)

# compute dataset statistic
print "Total of", len(allClasses), "classes"
hist = np.zeros(len(allClasses))
for i in range(len(datasetLabels)):
	hist[allClasses.index(datasetLabels[i])] += 1

print "min = ", min(hist)
print "max = ", max(hist)

width = 10
fig = plt.figure(num=None, figsize=(50, 6), dpi=80, facecolor='w', edgecolor='k')
x = np.arange(1, len(allClasses)+1)*10
bar1 = plt.bar(x, hist, width, color="y" )
plt.ylabel( '# occurrences' )
plt.xticks(x + width/2.0, allClasses )
plt.savefig("hist_MASTIF.png")
