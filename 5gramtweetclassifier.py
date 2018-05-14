import os
import re
import sys
import pickle
import xml.etree.ElementTree
import confusionmatrix

os.environ["CORENLP_HOME"] = "/usr/share/stanford-corenlp-full/"

from propertyreader import load_properties

from hyperdimensionalsemanticspace import SemanticSpace
from distutils.util import strtobool
import stringsequencespace
import sparsevectors
from logger import logger
import squintinglinguist


"""
This program takes PAN files and runs them against precomputed vectors.
"""


properties = load_properties("5gramtweetclassifier.properties")
corenlphome = str(properties["CORENLP_HOME"])
dimensionality = int(properties["dimensionality"])
denseness = int(properties["denseness"])
debug = bool(strtobool(properties["debug"]))
monitor = bool(strtobool(properties["monitor"]))
error = bool(strtobool(properties["error"]))
itempooldepth = int(properties["itempooldepth"])
averagelinkage = bool(strtobool(properties["averagelinkage"]))
maxlinkage = bool(strtobool(properties["maxlinkage"]))
votelinkage = bool(strtobool(properties["votelinkage"]))
categorymodelfilename = str(properties["categorymodelfilename"])
charactervectorspacefilename = str(properties["charactervectorspacefilename"])
resourcedirectory = str(properties["resourcedirectory"])
filenamepattern = str(properties["filenamepattern"])
frequencyweighting = bool(strtobool(properties["frequencyweighting"]))
fulltext = bool(strtobool(properties["fulltext"]))
generalise = bool(strtobool(properties["generalise"]))
featurise = bool(strtobool(properties["featurise"]))
postriples = bool(strtobool(properties["postriples"]))
postriplefile = str(properties["postriplefile"])
language = str(properties["language"])
outputdirectory = str(properties["outputdirectory"])
testbatchsize = int(properties["testbatchsize"])
window = int(properties["window"])

#os.environ["CORENLP_HOME"] = corenlphome


stringspace = stringsequencespace.StringSequenceSpace(dimensionality, denseness, window)
stringspace.importelementspace(charactervectorspacefilename)
stringspace.importpospermutations(postriplefile)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        resourcedirectory = sys.argv[1] + "/" + language + "/text/"
    if len(sys.argv) > 2:
        outputdirectory = sys.argv[2] + "/" + language + "/"

logger("input from: " + resourcedirectory, monitor)
logger("output to: " + outputdirectory, monitor)
if not os.path.exists(outputdirectory):
    os.makedirs(outputdirectory)

# create new vectors for test set

filenamelist = []
try:
    for filename in os.listdir(resourcedirectory):
        hitlist = re.match(filenamepattern, filename)
        if hitlist:  # and random.random() > 0.5:
            filenamelist.append(os.path.join(resourcedirectory, filename))
except:
    logger("No files found in " + resourcedirectory, error)

logger("Start building vectors for " + str(len(filenamelist)) + " test files.", monitor)
authorindex = 0
testvectors = {}
testitemspace = SemanticSpace()
decentfeatures = squintinglinguist.mediocremale + squintinglinguist.goodgenderones + squintinglinguist.mediocrefemale
nn = 0
for file in filenamelist:
    authorname = file.split(".")[0].split("/")[-1]
    authorindex += 1
    logger("Reading " + str(authorindex) + " " + file, monitor)
    workingvector = sparsevectors.newemptyvector(dimensionality)
    e = xml.etree.ElementTree.parse(file).getroot()
    hit = False
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
                if feature in decentfeatures:
                    hit = True
                    fv = stringspace.getvector(feature)
                    avector = sparsevectors.sparseadd(avector, sparsevectors.normalise(fv),
                                              stringspace.frequencyweight(feature))

        if postriples:
            posttriplevector = stringspace.postriplevector(origtext)
            avector = sparsevectors.sparseadd(avector, sparsevectors.normalise(posttriplevector))
        workingvector = sparsevectors.sparseadd(workingvector, sparsevectors.normalise(avector))
    nn += 1
    if not hit:
        logger(authorname + " did not get representation!", error)
    testitemspace.additem(authorindex, workingvector)
    testitemspace.name[authorindex] = authorname
logger("Done building " + str(nn) + " vectors.", monitor)

categories = set()
targetspace = SemanticSpace()
logger("Reading from " + categorymodelfilename, monitor)
cannedindexvectors = open(categorymodelfilename, "rb")
goingalong = True
n = 0
facit = {}
while goingalong:
    try:
        itemj = pickle.load(cannedindexvectors)
        targetspace.additem(itemj["authorindex"], itemj["vector"])
        targetspace.category[itemj["authorindex"]] = itemj["category"]
        targetspace.name[itemj["authorindex"]] = itemj["authorname"]
        categories.add(itemj["category"])
        if itemj["authorname"] in testitemspace.name.values():
            facit[itemj["authorname"]] = itemj["category"]
        n += 1
    except EOFError:
        goingalong = False
logger("Read " + str(n) + " target vectors.", monitor)
logger("Testing targetspace with " + str(len(categories)) + " categories, " + str(len(testitemspace.items()))
       + " test items and " +
       str(len(targetspace.items())) + " target items. ", monitor)
logger("Pool depth " + str(itempooldepth), monitor)
if averagelinkage:
    logger("Averagelinkage", monitor)
if votelinkage:
    logger("Votelinkage", monitor)

confusionmatrix = confusionmatrix.ConfusionMatrix()
nn = 0
for item in testitemspace.items():
    nn += 1
    logger("Testing\t" + str(nn) + "\t" + testitemspace.name[item], monitor)
    neighbours = {}
    for otheritem in targetspace.items():
        if testitemspace.name[item] == targetspace.name[otheritem]:
            continue
        neighbours[otheritem] = sparsevectors.sparsecosine(testitemspace.indexspace[item], targetspace.indexspace[otheritem])
    targetscore = {}
    sortedneighbours = sorted(neighbours,
                              key=lambda hh: neighbours[hh], reverse=True)[:itempooldepth]
    for target in categories:
        targetscore[target] = 0
    if averagelinkage:  # take all test neighbours and sum their scores
        for neighbour in sortedneighbours:
            targetscore[targetspace.category[neighbour]] += neighbours[neighbour]
    elif maxlinkage:    # use only the closest neighbour's score
        for neighbour in sortedneighbours:
            if targetscore[targetspace.category[neighbour]] < neighbours[neighbour]:
                targetscore[targetspace.category[neighbour]] = neighbours[neighbour]
    elif votelinkage:
        for neighbour in sortedneighbours:
            targetscore[targetspace.category[neighbour]] += 1
    sortedpredictions = sorted(categories, key=lambda ia: targetscore[ia], reverse=True)
    prediction = sortedpredictions[0]
    logger("Tested " + testitemspace.name[item] + ": " + prediction + " " + facit[testitemspace.name[item]], debug)
    confusionmatrix.addconfusion(facit[testitemspace.name[item]], prediction)
    with open(outputdirectory + testitemspace.name[item] + ".xml", "w") as outfile:
        print("<author id=\"" + testitemspace.name[item] + "\"", file=outfile)
        print("        lang=\"en\"", file=outfile)
        print("        gender_txt=\"" + prediction + "\"", file=outfile)
        print("/>", file=outfile)
        logger("Run output to " + outputdirectory + testitemspace.name[item] + ".xml", monitor)
logger("Done testing files.", monitor)
confusionmatrix.evaluate()

