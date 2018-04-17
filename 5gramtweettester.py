from hyperdimensionalsemanticspace import SemanticSpace
from propertyreader import load_properties
from distutils.util import strtobool
import pickle
import random
from logger import logger
from confusionmatrix import ConfusionMatrix

"""
This code reads precomputed vectors and does cross evaluation on them. Properties in separate file.
"""

properties = load_properties("5gramtweetxval.properties")
debug = bool(strtobool(properties["debug"]))
monitor = bool(strtobool(properties["monitor"]))
error = bool(strtobool(properties["error"]))
testtrainfraction = float(properties["testtrainfraction"])
testbatchsize = int(properties["testbatchsize"])
itempooldepth = int(properties["itempooldepth"])
averagelinkage = bool(strtobool(properties["averagelinkage"]))
maxlinkage = bool(strtobool(properties["maxlinkage"]))
votelinkage = bool(strtobool(properties["votelinkage"]))
categorymodelfilename = str(properties["categorymodelfilename"])
numberofiterations = int(properties["numberofiterations"])


def importvectors(filename):
    logger("Reading from " + filename, monitor)
    cannedindexvectors = open(filename, "rb")
    goingalong = True
    n = 0
    while goingalong:
        try:
            itemj = pickle.load(cannedindexvectors)
            itemspace.additem(itemj["authorindex"], itemj["vector"])
            itemspace.category[itemj["authorindex"]] = itemj["category"]
            itemspace.name[itemj["authorindex"]] = itemj["authorname"]
            categories.add(itemj["category"])
            n += 1
            if n > testbatchsize:
                break
        except EOFError:
            goingalong = False
    return n

categories = set()
itemspace = SemanticSpace()
importvectors(categorymodelfilename)
logger("Testing targetspace with " + str(len(categories)) + " categories, " + str(testbatchsize) +
       " test items. ", monitor)

iter = 0
for iterations in range(numberofiterations):
    iter += 1
    logger("Iteration " + str(iter), monitor)
    items = list(itemspace.items())
    if testtrainfraction > 0:
        random.shuffle(items)
        split = int(len(items) * testtrainfraction)
        testvectors = items[:split]
        knownvectors = items[split:]
    else:
        testvectors = items
        knownvectors = items
    logger("Calculating neighbours for " + str(len(testvectors)) +
           " test items and " + str(len(knownvectors)) + " target items.", monitor)
    neighbours = {}
    for item in testvectors:
        neighbours[item] = {}
        for otheritem in knownvectors:
            neighbours[item][otheritem] = itemspace.similarity(item, otheritem)
    logger("Done calculating neighbours", monitor)

    relativeneighbourhood = False
    if relativeneighbourhood:
        print("\t", end="\t")
        for item in items:
            print(item, end="\t")
        print("\n")
        for item in testvectors:
            print(item, end="\t")
            for otheritem in items:
                print(neighbours[item][otheritem], end="\t")
            print("\n")
        for item in items:
            print(str(item), itemspace.name[item])

    result = {}
    for c in categories:
        result[c] = {}

    for itempooldepth in [1, 11]:
        logger("Pool depth " + str(itempooldepth), monitor)
        if averagelinkage:
            logger("Averagelinkage", monitor)
        if votelinkage:
            logger("Votelinkage", monitor)
        confusion = ConfusionMatrix()
        targetscore = {}
        for item in testvectors:
            sortedneighbours = sorted(neighbours[item], key=lambda hh: neighbours[item][hh], reverse=True)[:itempooldepth]
            for target in categories:
                targetscore[target] = 0
            if averagelinkage:  # take all test neighbours and sum their scores
                for neighbour in sortedneighbours:
                    targetscore[itemspace.category[neighbour]] += neighbours[item][neighbour]
            elif maxlinkage:    # use only the closest neighbour's score
                for neighbour in sortedneighbours:
                    if targetscore[itemspace.category[neighbour]] < neighbours[item][neighbour]:
                        targetscore[itemspace.category[neighbour]] = neighbours[item][neighbour]
            elif votelinkage:
                for neighbour in sortedneighbours:
                    targetscore[itemspace.category[neighbour]] += 1
            sortedpredictions = sorted(categories, key=lambda ia: targetscore[ia], reverse=True)
            prediction = sortedpredictions[0]
            confusion.addconfusion(itemspace.category[item], prediction)
        confusion.evaluate()
        for c in categories:
            try:
                result[c][itempooldepth] = confusion.carat[c] / confusion.weight[c]
            except KeyError:
                result[c][itempooldepth] = 0
    logger("Done testing.", monitor)
    for c in categories:
        print(c, result[c], sep="\t")
