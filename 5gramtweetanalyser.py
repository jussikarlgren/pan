from hyperdimensionalsemanticspace import SemanticSpace
import json
import xml.etree.ElementTree
from stringsequencespace import StringSequenceSpace

import random
import os
import re

import sparsevectors
from logger import logger
from confusionmatrix import ConfusionMatrix


def tweetvector(string):
    uvector = sparsevectors.newemptyvector(dimensionality)
    if window > 0:
        windows = [string[ii:ii + window] for ii in range(len(string) - window + 1)]
        for sequence in windows:
            if ngramspace.contains(sequence):
                thisvector = ngramspace.indexspace[sequence]
#                 ngramspace.observe(sequence)  # should we be learning stuff now? naaw.
            else:
                thisvector = stringspace.makevector(sequence)
#                ngramspace.additem(sequence, thisvector)  # should it be added to cache? naaw.
            factor = ngramspace.frequencyweight(sequence)
            uvector = sparsevectors.sparseadd(uvector, sparsevectors.normalise(thisvector), factor)
    return uvector


def readgender(genderfile):
    global facittable
    facittable = {}
    with open(genderfile) as gf:
        for gline in gf:
            authoritem = gline.rstrip().split(":::")
            facittable[authoritem[0]] = authoritem[1]


targetlabel = ""
testvectors = {}
trainvectors = {}
window = 5
categories = ["male", "female"]
authornametable = {}
targets = set()
targetspace = {}
authorspace = {}  # can probably take this out
ngramspace = SemanticSpace()
ngramspace.addoperator("sequence")
confusion = ConfusionMatrix()
categorytable = {}
dimensionality = 2000

stringspace = StringSequenceSpace(ngramspace.dimensionality, ngramspace.denseness)

debug = False
monitor = True
error = True

testtrainfraction = 0.1
testbatchsize = 100
itempooldepth = 5  # keep this odd to avoid annoying ties
authorcategorisation = True
gendercategorisation = False
textcategorisation = False
averagelinkage = True
maxlinkage = False
frequencythreshold = 100

wordspacefile = "/home/jussi/data/wordspaces/pan18-5gram.fix.sorted.new.wordspace"
wordstatsfile = "/home/jussi/data/wordspaces/pan18-5gram.fix.sorted.new.wordstats"
resourcedirectory = "/home/jussi/data/pan/pan18-author-profiling-training-2018-02-27/en/text/"
genderfacitfilename = "/home/jussi/data/pan/pan18-author-profiling-training-2018-02-27/en/en.txt"
filenamepattern = ".+\.xml"
filenamelist = []
for filename in os.listdir(resourcedirectory):
    hitlist = re.match(filenamepattern, filename)
    if hitlist:  # and random.random() > 0.5:
        filenamelist.append(os.path.join(resourcedirectory, filename))

logger("Reading categories from facit file ", monitor)
readgender(genderfacitfilename)

random.shuffle(filenamelist)
logger("Setting off with a file list of " + str(len(filenamelist)) + " items.", monitor)


# batch = 61881  # 31881 covers up to frequency 100; # 6630 is up frequency 500; set to zero for full set
logger("Reading frequencies from " + wordstatsfile, monitor)
ngramspace.importstats(wordstatsfile)
cachevectors = False
if cachevectors:
    logger("Reading vectors from " + wordspacefile, monitor)
    (n1, n2) = ngramspace.importindexvectors(wordspacefile, frequencythreshold)
    logger("Imported " + n1 + " entirely new vectors and " + n2 + " previously known ones.", monitor)

if len(filenamelist) > testbatchsize:
    random.shuffle(filenamelist)  # if we shuffle here the weights won't be as good i mean overtrained
    filenamelist = filenamelist[:testbatchsize]
logger("Going on with a file list of " + str(testbatchsize) + " items.", monitor)

if textcategorisation:
    logger("Text target space", monitor)
if authorcategorisation:
    logger("Author target space", monitor)
if gendercategorisation:
    logger("Gender target space", monitor)
    for cat in categories:
        categorytable[cat] = cat  # redundant redundancy redundanciness
        targetspace[cat] = sparsevectors.newemptyvector(dimensionality)
        targets.add(cat)

logger("Started training files.", monitor)
authorindex = 0
textindex = 0
testvectorantal = 0
trainvectorantal = 0
for file in filenamelist:
    authorindex += 1
    authornametable[authorindex] = file.split(".")[0].split("/")[-1]
    logger("Starting training " + str(authorindex) + " " + file, debug)
    e = xml.etree.ElementTree.parse(file).getroot()
    authorspace[authorindex] = {}
    trainvectors[authorindex] = []
    testvectors[authorindex] = []
    thesevectors = []
    if authorcategorisation:
        targets.add(authorindex)
        targetlabel = authorindex
        targetspace[authorindex] = sparsevectors.newemptyvector(dimensionality)
        categorytable[authorindex] = facittable[authornametable[authorindex]]
    if gendercategorisation:
        targetlabel = facittable[authornametable[authorindex]]
    for b in e.iter("document"):
        textindex += 1
        if textcategorisation:
            targets.add(textindex)
            targetlabel = textindex
            targetspace[textindex] = sparsevectors.newemptyvector(dimensionality)
            categorytable[textindex] = facittable[authornametable[authorindex]]  # name space collision for keys
        avector = tweetvector(b.text)
        thesevectors.append((targetlabel, avector))

    if len(thesevectors) > 0:
        random.shuffle(thesevectors)
        split = int(len(thesevectors) * testtrainfraction)
        testvectors[authorindex] = thesevectors[:split]
        testvectorantal += len(testvectors[authorindex])
        trainvectors[authorindex] = thesevectors[split:]
        trainvectorantal += len(trainvectors[authorindex])
        for tv in trainvectors[authorindex]:
            targetspace[tv[0]] = sparsevectors.sparseadd(targetspace[tv[0]], tv[1])
logger("Done training files.", monitor)

logger("Testing targetspace with " + str(len(targetspace)) + " categories, " + str(testvectorantal) +
       " test items and " + str(trainvectorantal) +
       " training cases. ", monitor)
logger("Average linkage: " + str(averagelinkage) + " pool depth " + str(itempooldepth), monitor)
for authorindex in testvectors:
    logger(str(authorindex) + "\t" + str(facittable[authornametable[authorindex]]) + "===============", debug)
    targetscore = {}
    for target in targets:
        targetscore[target] = 0
    for testfile in testvectors[authorindex]:
        if averagelinkage:  # take all test sentences and sum their scores
            for target in targets:
                targetscore[target] += sparsevectors.sparsecosine(targetspace[target], testfile[1])
        elif maxlinkage:    # use only the closest sentence to match scores
            for target in targets:
                a = sparsevectors.sparsecosine(targetspace[target], testfile[1])
                if a > targetscore[target]:
                    targetscore[target] = a
    sortedtargets = sorted(targets, key=lambda ia: targetscore[ia], reverse=True)
    targetvote = {}
    for target in targets:
        for cat in categories:
            targetvote[cat] = 0
    for pp in sortedtargets[:itempooldepth]:
        targetvote[categorytable[pp]] += 1
        logger(str(pp) + "\t" + str(categorytable[pp]) + "\t" + str(targetscore[pp]), debug)
    sortedpredictions = sorted(categories, key=lambda ia: targetvote[ia], reverse=True)
    prediction = sortedpredictions[0]
    confusion.addconfusion(facittable[authornametable[authorindex]], prediction)

confusion.evaluate()