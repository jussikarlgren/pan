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
            if n >= testbatchsize:
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
resultaggregator = []
prunedresultaggregator = []
for iterations in range(numberofiterations):
    iter += 1
    logger("Iteration " + str(iter) + " of " + str(numberofiterations) + ".", monitor)
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

    cleanup = False  # this has now been shown not to be a good thing
    cleanuppooldepth = 11
    prunedknownvectors = []
    if cleanup:
        logger("Pruning.", monitor)
        throwout = {}
        friendstats = {}
        for c in categories:
            throwout[c] = 0
            friendstats[c] = 0
        nn = 0
        for candidate in knownvectors:
            nn += 1
            thiscategory = itemspace.category[candidate]
            friends = {}
            for friend in knownvectors:
                if not friend == candidate:
                    friends[friend] = itemspace.similarity(candidate, friend)
            sortedfriends = sorted(friends, key=lambda hh: friends[hh], reverse=True)[:cleanuppooldepth]
            whatitis = {}
            for c in categories:
                whatitis[c] = 0
            for potentialfriend in sortedfriends:
                whatitis[itemspace.category[potentialfriend]] += 1
            if whatitis[thiscategory] > cleanuppooldepth / 3:
                prunedknownvectors.append(candidate)
            else:
                throwout[thiscategory] += 1
                friendstats[thiscategory] += whatitis[thiscategory]
                logger(str(nn) + "\tThrew ut a " + str(thiscategory) +
                       " with " + str(whatitis[thiscategory]) + " correct friends.",
                       debug)
#        knownvectors = newknownvectors
        logger("Pruned targets to " + str(len(prunedknownvectors)) + " items.", monitor)
        for c in categories:
            qq = 0
            if friendstats[c] > 0:
                qq = throwout[c] / friendstats[c]
            logger(c + " " + str(throwout[c]) + " " + str(qq), monitor)



    neighbours = {}
    prunedneighbours = {}
    for item in testvectors:
        neighbours[item] = {}
        prunedneighbours[item] = {}
        for otheritem in knownvectors:
            neighbours[item][otheritem] = itemspace.similarity(item, otheritem)
            if otheritem in prunedknownvectors:
                prunedneighbours[item][otheritem] = neighbours[item][otheritem]
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
    prunedresult = {}
    for c in categories:
        result[c] = {}
        prunedresult[c] = {}

    logger("Pool depth " + str(itempooldepth), monitor)
    if averagelinkage:
        logger("Averagelinkage", monitor)
    if votelinkage:
        logger("Votelinkage", monitor)
    confusion = ConfusionMatrix()
    prunedconfusion = ConfusionMatrix()
    targetscore = {}
    prunedtargetscore = {}
    for item in testvectors:
        sortedneighbours = sorted(neighbours[item], key=lambda hh: neighbours[item][hh], reverse=True)[:itempooldepth]
        if cleanup:
            prunedsortedneighbours = sorted(prunedneighbours[item], key=lambda hh: prunedneighbours[item][hh], reverse=True)[:itempooldepth]
        for target in categories:
            targetscore[target] = 0
            prunedtargetscore[target] = 0
        if averagelinkage:  # take all test neighbours and sum their scores
            for neighbour in sortedneighbours:
                targetscore[itemspace.category[neighbour]] += neighbours[item][neighbour]
            if cleanup:
                for neighbour in prunedsortedneighbours:
                    prunedtargetscore[itemspace.category[neighbour]] += prunedneighbours[item][neighbour]
        elif maxlinkage:    # use only the closest neighbour's score
            for neighbour in sortedneighbours:
                if targetscore[itemspace.category[neighbour]] < neighbours[item][neighbour]:
                    targetscore[itemspace.category[neighbour]] = neighbours[item][neighbour]
        elif votelinkage:
            for neighbour in sortedneighbours:
                targetscore[itemspace.category[neighbour]] += 1
        sortedpredictions = sorted(categories, key=lambda ia: targetscore[ia], reverse=True)
        if cleanup:
            prunedsortedpredictions = sorted(categories, key=lambda ia: prunedtargetscore[ia], reverse=True)
        prediction = sortedpredictions[0]
        if cleanup:
            prunedprediction = prunedsortedpredictions[0]
        confusion.addconfusion(itemspace.category[item], prediction)
        if cleanup:
            prunedconfusion.addconfusion(itemspace.category[item], prunedprediction)
    confusion.evaluate()
    if cleanup:
        prunedconfusion.evaluate()
    for c in categories:
        try:
            result[c][itempooldepth] = confusion.carat[c] / confusion.weight[c]
            prunedresult[c][itempooldepth] = prunedconfusion.carat[c] / prunedconfusion.weight[c]
        except KeyError:
            result[c][itempooldepth] = 0
            prunedresult[c][itempooldepth] = 0
logger("Done testing.", monitor)
for c in categories:
    print(c, result[c], sep="\t")
    if cleanup:
        print(c, prunedresult[c], sep="\t")
    resultaggregator.append((c, result[c]))
    if cleanup:
        prunedresultaggregator.append((c, prunedresult[c]))

print(str(resultaggregator))
if cleanup:
    print(str(prunedresultaggregator))