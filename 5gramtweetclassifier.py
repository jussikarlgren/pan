from hyperdimensionalsemanticspace import SemanticSpace
from propertyreader import load_properties
from distutils.util import strtobool
import pickle
import stringsequencespace
import sparsevectors
from logger import logger
from confusionmatrix import ConfusionMatrix
import os
import re
import xml.etree.ElementTree


properties = load_properties("5gramtweetnewtester.properties")
dimensionality = int(properties["dimensionality"])
denseness = int(properties["denseness"])
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
charactervectorspacefilename = str(properties["charactervectorspacefilename"])
numberofiterations = int(properties["numberofiterations"])
resourcedirectory = str(properties["resourcedirectory"])
filenamepattern = str(properties["filenamepattern"])


stringspace = stringsequencespace.StringSequenceSpace()
stringspace.importcharacterspace(charactervectorspacefilename)

# create new vectors for test set

filenamelist = []
for filename in os.listdir(resourcedirectory):
    hitlist = re.match(filenamepattern, filename)
    if hitlist:  # and random.random() > 0.5:
        filenamelist.append(os.path.join(resourcedirectory, filename))


logger("Start building vectors for " + str(len(filenamelist)) + " test files.", monitor)
authorindex = 0
testvectors = {}
testitemspace = SemanticSpace()
nn = 0
for file in filenamelist:
    authorname = file.split(".")[0].split("/")[-1]
    authorindex += 1
    logger("Reading " + str(authorindex) + " " + file, monitor)
    workingvector = sparsevectors.newemptyvector(dimensionality)
#    modelitem = {}
#    modelitem["authorname"] = authorname
    e = xml.etree.ElementTree.parse(file).getroot()
    for b in e.iter("document"):
        avector = stringspace.textvector(b.text)
        workingvector = sparsevectors.sparseadd(workingvector, avector)
#    modelitem["vector"] = workingvector
    nn += 1
    if nn > testbatchsize:
        break
    testitemspace.additem(authorindex, workingvector)
    testitemspace.name[authorindex] = authorname
#    testvectors[authorindex] = modelitem
logger("Done building " + str(nn) + " vectors.", monitor)


categories = set()
targetspace = SemanticSpace()
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
        n += 1
    except EOFError:
        goingalong = False
logger("Read " + str(n) + " target vectors.", monitor)
logger("Testing targetspace with " + str(len(categories)) + " categories, " + str(len(testitemspace.items()))
       + " test items and " +
       str(len(targetspace.items())) + " target items. ", monitor)

neighbours = {}
for item in testitemspace.items():
    neighbours[item] = {}
    for otheritem in targetspace.items():
        if testitemspace.name[item] == targetspace.name[otheritem]:
            continue
        neighbours[item][otheritem] = targetspace.similarity(item, otheritem)
logger("Done calculating neighbours", monitor)

relativeneighbourhood = False
if relativeneighbourhood:
    print("\t", end="\t")
    for item in testitemspace.items():
        print(item, end="\t")
    print("\n")
    for item in testitemspace.items():
        print(item, end="\t")
        for otheritem in testitemspace.items():
            print(neighbours[item][otheritem], end="\t")
        print("\n")
    for item in testitemspace.items():
        print(str(item), targetspace.name[item])

result = {}
for c in categories:
    result[c] = {}

for iterations in range(numberofiterations):
    for itempooldepth in [1, 11]:
        logger("Pool depth " + str(itempooldepth), monitor)
        if averagelinkage:
            logger("Averagelinkage", monitor)
        if votelinkage:
            logger("Votelinkage", monitor)
        confusion = ConfusionMatrix()
        targetscore = {}
        for item in testitemspace.items():
            sortedneighbours = sorted(neighbours[item],
                                      key=lambda hh: neighbours[item][hh], reverse=True)[:itempooldepth]
            for target in categories:
                targetscore[target] = 0
            if averagelinkage:  # take all test neighbours and sum their scores
                for neighbour in sortedneighbours:
                    targetscore[targetspace.category[neighbour]] += neighbours[item][neighbour]
            elif maxlinkage:    # use only the closest neighbour's score
                for neighbour in sortedneighbours:
                    if targetscore[targetspace.category[neighbour]] < neighbours[item][neighbour]:
                        targetscore[targetspace.category[neighbour]] = neighbours[item][neighbour]
            elif votelinkage:
                for neighbour in sortedneighbours:
                    targetscore[targetspace.category[neighbour]] += 1
            sortedpredictions = sorted(categories, key=lambda ia: targetscore[ia], reverse=True)
            prediction = sortedpredictions[0]
            confusion.addconfusion(targetspace.category[item], prediction)
        confusion.evaluate()
        for c in categories:
            result[c][itempooldepth] = confusion.carat[c] / confusion.weight[c]
logger("Done testing files.", monitor)

for c in categories:
    print(c, result[c], sep="\t")
