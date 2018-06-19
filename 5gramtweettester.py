import os
import re
import xml

os.environ["CORENLP_HOME"] = "/usr/share/stanford-corenlp-full/"

import sparsevectors
import squintinglinguist
import stringsequencespace
from hyperdimensionalsemanticspace import SemanticSpace
from propertyreader import load_properties
from distutils.util import strtobool
import pickle
import random
from logger import logger
from confusionmatrix import ConfusionMatrix

"""
This code reads precomputed vectors and some other files and tests various approaches for encoding 
visavis the precomputed vectors. Properties in separate file.
"""

properties = load_properties("5gramtweettester.properties")
corenlp = properties["CORENLP_HOME"]
debug = bool(strtobool(properties["debug"]))
monitor = bool(strtobool(properties["monitor"]))
error = bool(strtobool(properties["error"]))
testtrainfraction = float(properties["testtrainfraction"])
testbatchsize = int(properties["testbatchsize"])
itempooldepth = int(properties["itempooldepth"])
averagelinkage = bool(strtobool(properties["averagelinkage"]))
maxlinkage = bool(strtobool(properties["maxlinkage"]))
votelinkage = bool(strtobool(properties["votelinkage"]))

dimensionality = int(properties["dimensionality"])
denseness = int(properties["denseness"])
categorymodelfilename = str(properties["categorymodelfilename"])
charactervectorspacefilename = str(properties["charactervectorspacefilename"])
genderfacitfilename = str(properties["genderfacitfilename"])
resourcedirectory = str(properties["resourcedirectory"])
filenamepattern = str(properties["filenamepattern"])
frequencyweighting = bool(strtobool(properties["frequencyweighting"]))
fulltext = bool(strtobool(properties["fulltext"]))
generalise = bool(strtobool(properties["generalise"]))
featurise = bool(strtobool(properties["featurise"]))
postriples = bool(strtobool(properties["postriples"]))
postriplefile = str(properties["postriplefile"])
window = int(properties["window"])
cycles = int(properties["cycles"])
unknownbatchsize = int(properties["unknownbatchsize"])


stringspace = stringsequencespace.StringSequenceSpace(dimensionality, denseness, window)
stringspace.importelementspace(charactervectorspacefilename)

categories = set()
targetspace = SemanticSpace()


logger("Reading categories from facit file ", monitor)
facittable = {}
with open(genderfacitfilename) as gf:
    for gline in gf:
        authoritem = gline.rstrip().split(":::")
        facittable[authoritem[0]] = authoritem[1]



logger("Reading from " + categorymodelfilename, monitor)
cannedindexvectors = open(categorymodelfilename, "rb")
goingalong = True
n = 0
while goingalong:
    try:
        itemj = pickle.load(cannedindexvectors)
        targetspace.additem(itemj["authorindex"], itemj["vector"])
        targetspace.category[itemj["authorindex"]] = itemj["category"]
        targetspace.name[itemj["authorindex"]] = itemj["authorname"]
        categories.add(itemj["category"])
        if n >= testbatchsize:
            break
        n += 1
    except EOFError:
        goingalong = False
testbatchsize = n

logger("Testing targetspace with " + str(len(categories)) + " categories, " + str(testbatchsize) +
       " test items. ", monitor)

# create new vectors for test set

filenamelist = []
for filename in os.listdir(resourcedirectory):
    hitlist = re.match(filenamepattern, filename)
    if hitlist:  # and random.random() > 0.5:
        filenamelist.append(os.path.join(resourcedirectory, filename))

if testtrainfraction > 0:
    random.shuffle(filenamelist)
    split = int(len(filenamelist) * testtrainfraction)
    testfiles = filenamelist[:split]
else:
    testfiles = filenamelist

logger("Start building vectors for " + str(len(testfiles)) + " test files.", monitor)
authorindex = 0
testitemspace = SemanticSpace()
nn = 0
for file in testfiles:
    authorname = file.split(".")[0].split("/")[-1]
    authorindex += 1
    logger("Reading " + str(authorindex) + " " + file, monitor)
    workingvector = sparsevectors.newemptyvector(dimensionality)
    e = xml.etree.ElementTree.parse(file).getroot()

    for b in e.iter("document"):
        origtext = b.text
        avector = sparsevectors.newemptyvector(dimensionality)
        if fulltext:
            avector = sparsevectors.normalise(stringspace.textvector(origtext, frequencyweighting))
        if generalise:
            newtext = squintinglinguist.generalise(origtext)
            avector = sparsevectors.sparseadd(avector,
                                              sparsevectors.normalise(stringspace.textvector(newtext,
                                                                                             frequencyweighting)))
        if featurise:
            features = squintinglinguist.featurise(origtext)
            for feature in features:
                fv = stringspace.getvector(feature)
                avector = sparsevectors.sparseadd(avector, sparsevectors.normalise(fv),
                                                  stringspace.frequencyweight(feature))
        if postriples:
            posttriplevector = stringspace.postriplevector(origtext)
            avector = sparsevectors.sparseadd(avector, sparsevectors.normalise(posttriplevector))
        workingvector = sparsevectors.sparseadd(workingvector, sparsevectors.normalise(avector))
    nn += 1
    testitemspace.additem(authorindex, workingvector)
    testitemspace.name[authorindex] = authorname
logger("Done building " + str(nn) + " vectors.", monitor)


for cycle in range(cycles):
    testitemnames = list(testitemspace.items())
    random.shuffle(testitemnames)
    testers = testitemnames[:unknownbatchsize]

    logger("Cycle " + str(cycle) + " of " + str(cycles) + "tests.")
    items = list(targetspace.items())
    logger("Calculating neighbours for " + str(len(testitemspace.items())) +
           " test items and " + str(len(targetspace.items())) + " target items.", monitor)

    neighbours = {}
    for item in testers:
        neighbours[item] = {}
        for otheritem in targetspace.items():
            if testitemspace.name[item] == targetspace.name[otheritem]:
                continue
            neighbours[item][otheritem] = sparsevectors.sparsecosine(testitemspace.indexspace[item], targetspace.indexspace[otheritem])
    logger("Done calculating neighbours", monitor)


    logger("Pool depth " + str(itempooldepth), monitor)
    if averagelinkage:
        logger("Averagelinkage", monitor)
    if votelinkage:
        logger("Votelinkage", monitor)
    confusion = ConfusionMatrix()
    primeconfusion = ConfusionMatrix()
    targetscore = {}
    for item in testers:
        sortedneighbours = sorted(neighbours[item], key=lambda hh: neighbours[item][hh], reverse=True)[:itempooldepth]
        primeconfusion.addconfusion(facittable[testitemspace.name[item]], targetspace.category[sortedneighbours[0]])
        for target in categories:
            targetscore[target] = 0
        if averagelinkage:  # take all test neighbours and sum their scores
            for neighbour in sortedneighbours:
                targetscore[targetspace.category[neighbour]] += neighbours[item][neighbour]
        elif votelinkage:
            for neighbour in sortedneighbours:
                targetscore[targetspace.category[neighbour]] += 1
        sortedpredictions = sorted(categories, key=lambda ia: targetscore[ia], reverse=True)
        prediction = sortedpredictions[0]
        logger(prediction + "?" + " " + facittable[testitemspace.name[item]] + ".", debug)
        for iii in range(itempooldepth):
            try:
                logger(testitemspace.name[item] + " (" + facittable[testitemspace.name[item]] + ") " +
                      "\t" + str(neighbours[item][sortedneighbours[iii]]) + "\t" + targetspace.name[sortedneighbours[iii]] +
                      " (" + targetspace.category[sortedneighbours[iii]] + ") "
                      , debug)
            except:
                logger("keyerror " + str(iii), error)
        confusion.addconfusion(facittable[testitemspace.name[item]], prediction)
    print("1","-----------")
    primeconfusion.evaluate()
    print(itempooldepth,"-----------")
    confusion.evaluate()
    recallresult = {}
    precisionresult = {}
    precisionresult["both"] = {}
    correct = {}
    correct[itempooldepth] = 0
    correct[1] = 0
    cands = {}
    cands[itempooldepth] = 0
    cands[1] = 0
    for c in categories:
        recallresult[c] = {}
        precisionresult[c] = {}
        try:
            cands[itempooldepth] += confusion.glitterweight[c]
            correct[itempooldepth] += confusion.carat[c]
            recallresult[c][itempooldepth] = confusion.carat[c] / confusion.weight[c]
            precisionresult[c][itempooldepth] = confusion.carat[c] / confusion.glitterweight[c]
        except KeyError:
            recallresult[c][itempooldepth] = 0
        try:
            cands[1] += primeconfusion.glitterweight[c]
            correct[1] += primeconfusion.carat[c]
            recallresult[c][1] = primeconfusion.carat[c] / primeconfusion.weight[c]
            precisionresult[c][1] = primeconfusion.carat[c] / primeconfusion.glitterweight[c]
        except KeyError:
            recallresult[c][1] = 0
    precisionresult["both"][itempooldepth] = correct[itempooldepth] / cands[itempooldepth]
    precisionresult["both"][1] = correct[1] / cands[1]
    logger("Done testing.", monitor)
    outputcategories = list(categories) + ["both"]
    print("Precision")
    for c in outputcategories:
        print(" ", "&", c, sep="\t", end="\t")
    for c in outputcategories:
        print(" ", "&", c, sep="\t", end="\t")
    print()
    for c in outputcategories:
        print(" ", "&", precisionresult[c][1], sep="\t", end="\t")
    for c in outputcategories:
        print(" ", "&", precisionresult[c][itempooldepth], sep="\t", end="\t")
    print()
    print("Recall")
    for c in categories:
        print(" ", "&", c, sep="\t", end="\t")
    for c in categories:
        print(" ", "&", c, sep="\t", end="\t")
    print()
    for c in categories:
        print(" ", "&", recallresult[c][1], sep="\t", end="\t")
    for c in categories:
        print(" ", "&", recallresult[c][itempooldepth], sep="\t", end="\t")
    print()
