import xml.etree.ElementTree

from stringsequencespace import StringSequenceSpace
import os
import re

from logger import logger

debug = True
monitor = True
error = True
wordstatsfile = "/home/jussi/data/wordspaces/pan18.5gram.newstats"
resourcedirectory = "/home/jussi/data/pan/pan18-author-profiling-training-2018-02-27/en/text/"
filenamepattern = "+\.xml"
dimensionality = 2000
window = 5
denseness = 0

targetlabel = ""
targets = set()
targetspace = {}
# ngramspace = SemanticSpace()
stringspace = StringSequenceSpace(dimensionality, denseness, window)
filenamelist = []
for filename in os.listdir(resourcedirectory):
    logger(filename, monitor)
#    hitlist = re.match(filenamepattern, filename)
#    if hitlist:  # and random.random() > 0.5:
    filenamelist.append(os.path.join(resourcedirectory, filename))

logger("Started training files.", monitor)
authorindex = 0
textindex = 0
n = 0

for file in filenamelist:
    authorindex += 1
    logger("Starting training " + str(authorindex) + " " + file, monitor)
    e = xml.etree.ElementTree.parse(file).getroot()
    for b in e.iter("document"):
        origtext = b.text
        stringspace.observe(origtext)
logger("Done training files resulting in " + str(n) + " vectors.", monitor)
logger("Frequencies saved to " + wordstatsfile, monitor)
stringspace.savefrequencies(wordstatsfile)