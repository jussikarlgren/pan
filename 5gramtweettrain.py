from hyperdimensionalsemanticspace import SemanticSpace
import xml.etree.ElementTree
from stringsequencespace import StringSequenceSpace
from propertyreader import load_properties
from distutils.util import strtobool
import pickle
import random
import os
import re

import sparsevectors
from logger import logger
from confusionmatrix import ConfusionMatrix

properties = load_properties("factory.properties")
dimensionality = int(properties["dimensionality"])
denseness = int(properties["denseness"])
window = int(properties["window"])
debug = bool(strtobool(properties["debug"]))
monitor = bool(strtobool(properties["monitor"]))
error = bool(strtobool(properties["error"]))
testbatchsize = int(properties["testbatchsize"])
authorcategorisation = bool(strtobool(properties["authorcategorisation"]))
gendercategorisation = bool(strtobool(properties["gendercategorisation"]))
textcategorisation = bool(strtobool(properties["textcategorisation"]))
frequencythreshold = int(properties["frequencythreshold"])
wordstatsfile = str(properties["wordstatsfile"])
resourcedirectory = str(properties["resourcedirectory"])
filenamepattern = str(properties["filenamepattern"])
genderfacitfilename = str(properties["genderfacitfilename"])
charactervectorspacefilename = str(properties["charactervectorspacefilename"])
categorymodelfilename = str(properties["categorymodelfilename"])


def readgender(genderfile):
    global facittable
    facittable = {}
    with open(genderfile) as gf:
        for gline in gf:
            authoritem = gline.rstrip().split(":::")
            facittable[authoritem[0]] = authoritem[1]


targetlabel = ""
targets = set()
targetspace = {}
# ngramspace = SemanticSpace()
stringspace = StringSequenceSpace(dimensionality, denseness, window)
categories = ["male", "female"]

filenamelist = []
for filename in os.listdir(resourcedirectory):
    hitlist = re.match(filenamepattern, filename)
    if hitlist:  # and random.random() > 0.5:
        filenamelist.append(os.path.join(resourcedirectory, filename))



logger("Reading categories from facit file ", monitor)
readgender(genderfacitfilename)

random.shuffle(filenamelist)
logger("Setting off with a file list of " + str(len(filenamelist)) + " items.", monitor)

logger("Reading frequencies from " + wordstatsfile, monitor)
stringspace.importstats(wordstatsfile)

if len(filenamelist) > testbatchsize:
    random.shuffle(filenamelist)  # if we shuffle here the weights won't be as good i mean overtrained
    filenamelist = filenamelist[:testbatchsize]
logger("Going on with a file list of " + str(len(filenamelist)) + " items.", monitor)

if textcategorisation:
    logger("Text target space", monitor)
if authorcategorisation:
    logger("Author target space", monitor)
if gendercategorisation:
    logger("Gender target space", monitor)
    catitem = {}
    for cat in categories:
        catitem[cat] = {}

logger("Started training files.", monitor)
authorindex = 0
textindex = 0
with open(categorymodelfilename, "wb") as outfile:
    for file in filenamelist:
        authorname = file.split(".")[0].split("/")[-1]
        authorindex += 1
        logger("Starting training " + str(authorindex) + " " + file, monitor)
        if authorcategorisation:
            workingvector = sparsevectors.newemptyvector(dimensionality)
            modelitem = {}
            modelitem["textindex"] = 0
            modelitem["authorindex"] = authorindex
            modelitem["authorname"] = authorname
            modelitem["category"] = facittable[authorname]
        e = xml.etree.ElementTree.parse(file).getroot()
        for b in e.iter("document"):
            avector = stringspace.textvector(b.text)
            if textcategorisation:
                textindex += 1
                modelitem = {}
                modelitem["textindex"] = textindex
                modelitem["authorindex"] = authorindex
                modelitem["authorname"] = authorname
                modelitem["category"] = facittable[authorname]
                modelitem["vector"] = avector
                pickle.dump(modelitem, outfile)
            if authorcategorisation:
                workingvector = sparsevectors.sparseadd(workingvector, avector)
            if gendercategorisation:
                catitem[facittable[authorname]] = sparsevectors.sparseadd(catitem[facittable[authorname]],
                                                                          avector)
        if authorcategorisation:
            modelitem["vector"] = workingvector
            pickle.dump(modelitem, outfile)
    if gendercategorisation:
        for cat in categories:
            modelitem = {}
            modelitem["category"] = cat
            modelitem["authorindex"] = "legion"
            modelitem["textindex"] = "legion"
            modelitem["authorname"] = "legion"
            modelitem["vector"] = catitem[cat]
            pickle.dump(modelitem, outfile)
logger("Done training files resulting in " + str(len(targetspace)) + " vectors.", monitor)
# output character patterns to be able to generate new tweetvectors for separate testing on trained data
logger("Saving character setup.", monitor)
stringspace.savecharacterspace(charactervectorspacefilename)

