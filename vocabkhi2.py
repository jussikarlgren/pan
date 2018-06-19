import os
import re
import sys
import pickle
import xml.etree.ElementTree
import confusionmatrix

os.environ["CORENLP_HOME"] = "/usr/share/stanford-corenlp-full/"

from hyperdimensionalsemanticspace import SemanticSpace
import stringsequencespace
import sparsevectors
from logger import logger
import squintinglinguist


# properties file. line must start with a "#" if it is a comment.
# no trailing comments: they will mess up.

# loglevel settings
debug = True
monitor = True
error = True

dimensionality = 2000
denseness = 10
window = 0


# where the data live
resourcedirectory = "/home/jussi/data/pan/pan18-author-profiling-training-2018-02-27/en/text/"
# resourcedirectory = "/home/jussi/data/pan/test/"
filenamepattern = "c.+\.xml"
language = "en"
# how many items in test set? there are 3000 max.
testbatchsize = 200
# keep this odd to avoid annoying ties
# parameter search seems to indicate this should be one
itempooldepth = 1
# one of the three following must be True, the other must be False
averagelinkage = True
maxlinkage = False
votelinkage = False
# these two files are where the saved trained model can be found
charactervectorspacefilename = "/home/jussi/data/wordspaces/factory.characters.author.weight.fgpp5"
categorymodelfilename = "/home/jussi/data/wordspaces/factory.author.weight.fgpp5"
postriplefile = "/home/jussi/data/wordspaces/pan18.postriples.fgpp5"
# frequency statistics for observable items
wordstatsfile = "/home/jussi/data/wordspaces/pan18.newstats"
CORENLP_HOME = "/usr/share/stanford-corenlp-full/"

frequencyweighting = "True"
fulltext = False
generalise = False
featurise = True
postriples = True

outputdirectory = "/home/jussi/data/pan/pan18-author-profiling-output/"




#os.environ["CORENLP_HOME"] = corenlphome


stringspace = stringsequencespace.StringSequenceSpace(dimensionality, denseness, window)
stringspace.importelementspace(charactervectorspacefilename)
stringspace.importpospermutations(postriplefile)
stringspace.importfrequencies(wordstatsfile)

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
authorvocab = {}
for file in filenamelist:
    authorname = file.split(".")[0].split("/")[-1]
    authorvocab[authorname] = set()
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
        authorvocab[authorname].update(squintinglinguist.words(origtext))
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

confusionwords = {}
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
    if facit[testitemspace.name[item]]+prediction not in confusionwords:
        confusionwords[facit[testitemspace.name[item]]+prediction] = set()
    confusionwords[facit[testitemspace.name[item]]+prediction].update(authorvocab[testitemspace.name[item]])
    with open(outputdirectory + testitemspace.name[item] + ".xml", "w") as outfile:
        print("<author id=\"" + testitemspace.name[item] + "\"", file=outfile)
        print("        lang=\"en\"", file=outfile)
        print("        gender_txt=\"" + prediction + "\"", file=outfile)
        print("/>", file=outfile)
        logger("Run output to " + outputdirectory + testitemspace.name[item] + ".xml", monitor)
logger("Done testing files.", monitor)
confusionmatrix.evaluate()



def except_keys(d, keys):
    dd = set()
    for x in d:
        if x not in keys:
            dd.update(d[x])
    return dd

frr = 100
for j in confusionwords:
    print(j)
    ddd = except_keys(confusionwords, [j])
    d3 = set()
    d2 = set()
    for dw in confusionwords[j]:
        try:
            if stringspace.globalfrequency[dw] > frr:
                d2.add(dw)
        except KeyError:
            print("***", dw)
    for dw in ddd:
        try:
            if stringspace.globalfrequency[dw] > frr:
                d3.add(dw)
        except KeyError:
            print("***", dw)

    print(d2.difference(d3))


