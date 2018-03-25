from hyperdimensionalsemanticspace import SemanticSpace
import sparsevectors
import xml.etree.ElementTree
import random
import os
import re
from logger import logger
import json
from confusionmatrix import ConfusionMatrix

def trainglobal(string):
    if window > 0:
        windows = [string[i:i + window] for i in range(len(string) - window + 1)]
        for sequence in windows:
            if ngramspace.contains(sequence):
                ngramspace.observe(sequence)
            else:
                thisvector = {}
                for character in sequence:
                    ngramspace.checkwordspace(character)
                    thisvector = sparsevectors.sparseadd(
                        sparsevectors.permute(thisvector, ngramspace.permutationcollection["sequence"]),
                            ngramspace.indexspace[character])
                ngramspace.additem(sequence, thisvector)


def tweetvector(string):
    uvector = {}
    if window > 0:
        windows = [string[i:i + window] for i in range(len(string) - window + 1)]
        for sequence in windows:
            if ngramspace.contains(sequence):
                thisvector = ngramspace.indexspace[sequence]
                ngramspace.observe(sequence)
            else:
                thisvector = {}
                for character in sequence:
                    ngramspace.checkwordspace(character)
                    thisvector = sparsevectors.sparseadd(
                        sparsevectors.permute(thisvector, ngramspace.permutationcollection["sequence"]),
                            ngramspace.indexspace[character])
                ngramspace.additem(sequence, thisvector)
            factor = ngramspace.frequencyweight(sequence)
            uvector = sparsevectors.sparseadd(uvector, sparsevectors.normalise(thisvector),factor)
    return uvector

def unittest():
    a = tweetvector("abcdefghijklmn")
    b = tweetvector("opqrstuvwxyzabcdef")
    print(sparsevectors.sparsecosine(a,b))

def readgender(genderfile):
    global gendertable
    gendertable = {}
    with open(genderfile) as gf:
        for line in gf:
            authoritem = line.rstrip().split(":::")
            gendertable[authoritem[0]] = authoritem[1]

authorspace = {}
testvectors = {}
trainvectors = {}
testtrainfraction = 0.1
debug = False
monitor = True
error = True
authorindex = {}
testbatchsize = 3000

window = 5
ngramspace = SemanticSpace()
ngramspace.addoperator("sequence")

confusion = ConfusionMatrix()

resourcedirectory = "/home/jussi/data/pan/pan18-author-profiling-training-2018-02-27/en/text/"

filenamepattern = ".+\.xml"
filenamelist = []
for filename in os.listdir(resourcedirectory):
    hitlist = re.match(filenamepattern, filename)
    if hitlist:  # and random.random() > 0.5:
        filenamelist.append(os.path.join(resourcedirectory, filename))

genderfacitfilename = "/home/jussi/data/pan/pan18-author-profiling-training-2018-02-27/en/en.txt"
readgender(genderfacitfilename)

random.shuffle(filenamelist)

logger("Setting off with a file list of " + str(len(filenamelist)) + " items.", monitor)
wordspacefile = "/home/jussi/data/wordspaces/pan18-5gram.fix.wordspace"
train = False
if train:
    batch = 50
    i = 0
    for file in filenamelist:
        i += 1
        logger("Training weights with " + str(i) + " files " + str(file), monitor)
        e = xml.etree.ElementTree.parse(file).getroot()
        for b in e.iter("document"):
            trainglobal(b.text)
        if i > batch:
            logger("Finished training with " + i + " files processed.", monitor)
            break
    logger("Writing weights to " + wordspacefile, monitor)
    ngramspace.outputwordspace(wordspacefile)
else:
    logger("Reading weights from " + wordspacefile, monitor)
    with open(wordspacefile) as savedwordspace:
        for line in savedwordspace:
            ngramspace.addsaveditem(json.loads(line))

if len(filenamelist) > testbatchsize:
    random.shuffle(filenamelist)  # if we shuffle here the weights won't be as good i mean overtrained
    filenamelist = filenamelist[:testbatchsize]

index = 0
for file in filenamelist:
    index += 1
    authorindex[index] = file.split(".")[0].split("/")[-1]
    logger("Starting training " + str(index) + " " + file, monitor)
    e = xml.etree.ElementTree.parse(file).getroot()
    authorspace[index] = {}
    trainvectors[index] = []
    testvectors[index] = []
    thesevectors = []
    for b in e.iter("document"):
        logger(file + "\t" + b.text, debug)
        thesevectors.append(tweetvector(b.text))
        authorspace[index] = sparsevectors.sparseadd(authorspace[index], tweetvector(b.text))
    if len(thesevectors) > 0:
        random.shuffle(thesevectors)
        split = int(len(thesevectors) * testtrainfraction)
        testvectors[index] = thesevectors[:split]
        trainvectors[index] = thesevectors[split:]

for index in authorindex:
    print("\t", end="")
    print(index, end="\t")
print()

for index in authorindex:
    print(index, end="\t")
    for otherindex in authorindex:
        print(sparsevectors.sparsecosine(authorspace[index], authorspace[otherindex]), sep="\t", end="\t")
    print()

logger("Testing authorspace ", monitor)
for index in authorindex:
    print(str(index), gendertable[authorindex[index]], "===============", sep="\t")
    authorscore = {}
    for author in authorspace:
        authorscore[author] = 0
        for testfile in testvectors[index]:
            authorscore[author] += sparsevectors.sparsecosine(authorspace[author], testfile)
    sortedauthors = sorted(authorspace, key=lambda ia: authorscore[ia], reverse=True)
    for author in sortedauthors:
        print(str(author), gendertable[authorindex[author]], str(authorscore[author]), sep="\t")
    confusion.addconfusion(index, sortedauthors[0])
    confusion.addconfusion(gendertable[authorindex[index]], gendertable[authorindex[sortedauthors[0]]])

confusion.evaluate()

textspace = False
if textspace:
    logger("Testing textspace ", monitor)
    for index in authorindex:
        print(str(index) + "===============")
        authorscore = {}
        for author in authorspace:
            authorscore[author] = []
            for testfile in testvectors[index]:
                for trainfile in trainvectors[author]:
                    authorscore[author].append(sparsevectors.sparsecosine(trainfile, testfile))
        for author in authorspace:
            print(str(author) + "\t" + str(sorted(authorscore[author], reverse=True)[:5]))
