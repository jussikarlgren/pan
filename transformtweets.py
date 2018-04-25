import xml.etree.ElementTree
from distutils.util import strtobool
import squintinglinguist
import os
import re
from logger import logger

debug = False
monitor = True
error = True

resourcedirectory = "/home/jussi/data/pan/pan18-author-profiling-training-2018-02-27/en/text/"
filenamepattern = ".+\.xml"
transformedfiledirectory = "/home/jussi/data/pan/pan18-author-profiling-training-2018-02-27/en/en-transformed-text/"

filenamelist = []
for filename in os.listdir(resourcedirectory):
    hitlist = re.match(filenamepattern, filename)
    if hitlist:  # and random.random() > 0.5:
        filenamelist.append(filename)

logger("Going forth with a file list of " + str(len(filenamelist)) + " items.", monitor)

# with open(transformedfilename, "wb") as outfile:
for filename in filenamelist:
    o = open(os.path.join(transformedfiledirectory, filename), "wb")
    e = xml.etree.ElementTree.parse(os.path.join(resourcedirectory, filename)).getroot()
    for b in e.iter("document"):
        b.text = squintinglinguist.generalise(b.text, True, True, True, True, False)
    o.write(xml.etree.ElementTree.tostring(e))
    o.close()

logger("New texts saved to " + transformedfiledirectory, monitor)
# output character patterns to be able to generate new tweetvectors for separate testing on trained data

