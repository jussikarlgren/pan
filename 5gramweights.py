import collections
import xml.etree.ElementTree
from stringsequencespace import StringSequenceSpace
import random
import os
import re
from logger import logger
import pickle

debug = False
monitor = True
error = True

wordspacefile = "/home/jussi/data/wordspaces/pan18-5gram.fix.sorted.new.wordspace"
wordstatsfile = "/home/jussi/data/wordspaces/pan18-5gram.fix.sorted.new.wordstats"
resourcedirectory = "/home/jussi/data/pan/pan18-author-profiling-training-2018-02-27/en/text/"

charspace = {}
dimensionality = 2000
denseness = 10
seen = collections.Counter()
stringspace = StringSequenceSpace(dimensionality, denseness)
window = 5



filenamepattern = ".+\.xml"
filenamelist = []
for filename in os.listdir(resourcedirectory):
    hitlist = re.match(filenamepattern, filename)
    if hitlist:  # and random.random() > 0.5:
        filenamelist.append(os.path.join(resourcedirectory, filename))

setsize = 3000
random.shuffle(filenamelist)
filenamelist = filenamelist[:setsize]

logger("Setting off with a file list of " + str(len(filenamelist)) + " items.", monitor)
i = 0
batch = 100
j = 0
k = 0
vectoroutfile = open(wordspacefile, 'wb')
logger("Writing vectors to " + wordspacefile, monitor)
for file in filenamelist:
    i += 1
    j += 1
    k += 1
    if j > batch:
        j = 0
        statsoutfile = open(wordstatsfile, 'w')
        logger("Writing " + str(sum(seen.values())) + " items to " + wordstatsfile, monitor)
        for something in seen:
            if seen[something] > 1:
                statsoutfile.write(something + "\t" + str(seen[something]) + "\n")
        statsoutfile.flush()
        statsoutfile.close()
    logger("Computing ngram frequency-based weights with " + str(i) + " files " + str(file), monitor)
    e = xml.etree.ElementTree.parse(file).getroot()
    for b in e.iter("document"):
        string = b.text
        windows = [string[ii:ii + window] for ii in range(len(string) - window + 1)]
        for sequence in windows:
            seen[sequence] += 1
            if seen[sequence] == 1:
                thisvector = stringspace.makevector(sequence)
                itemj = {}
                itemj["string"] = sequence
                itemj["indexvector"] = thisvector
                vectoroutfile.write(pickle.dumps(itemj, protocol=0))
                vectoroutfile.flush()


