import xml.etree.ElementTree
import nltk
import os
import re
from nltk.tokenize import word_tokenize
import random


from logger import logger

genderfacitfilename = "/home/jussi/data/pan/pan18-author-profiling-training-2018-02-27/en/en.txt"
def readgender(genderfile):
    global facittable
    facittable = {}
    with open(genderfile) as gf:
        for gline in gf:
            authoritem = gline.rstrip().split(":::")
            facittable[authoritem[0]] = authoritem[1]


categories = ["male", "female"]
authornametable = {}

debug = True
monitor = True
error = True

testbatchsize = 3000

resourcedirectory = "/home/jussi/data/pan/pan18-author-profiling-training-2018-02-27/en/text/"
filenamepattern = ".+\.xml"
filenamelist = []
for filename in os.listdir(resourcedirectory):
    hitlist = re.match(filenamepattern, filename)
    if hitlist:  # and random.random() > 0.5:
        filenamelist.append(os.path.join(resourcedirectory, filename))
random.shuffle(filenamelist)
filenamelist = filenamelist[:testbatchsize]

logger("Reading categories from facit file ", monitor)
readgender(genderfacitfilename)

logger("Setting off with a file list of " + str(len(filenamelist)) + " items.", monitor)
lexcats = ["noun", "verb", "adjective", "all"]

verbpos = ["VBZ", "VBP", "VBD", "VBN", "VBG", "VB"]
nounpos = ["NN", "NNP", "NNS", "NNPS"]
adjpos = ["JJ", "JJS", "JJR"]

tokens = {}
types = {}
bign = 0
for c in categories:
    types[c] = {}
    tokens[c] = {}
    for l in lexcats:
        types[c][l] = {}
        tokens[c][l] = 0

for file in filenamelist:
    author = file.split(".")[0].split("/")[-1]
    c = facittable[author]
    types[author] = {}
    tokens[author] = {}
    for l in lexcats:
        types[author][l] = {}
        tokens[author][l] = 0
    e = xml.etree.ElementTree.parse(file).getroot()
    for b in e.iter("document"):
        bign += 1
        text = word_tokenize(b.text)
        poses = nltk.pos_tag(text)
        for p in poses:
            w = p[0]
            l = p[1]
            theselexcats = ["all"]
            if l in nounpos:
                theselexcats.append("noun")
            elif l in verbpos:
                theselexcats.append("verb")
            elif l in adjpos:
                theselexcats.append("adjective")
            for lexcat in theselexcats:
                tokens[c][lexcat] += 1
                if w in types[c][lexcat]:
                    types[c][lexcat][w] += 1
                else:
                    types[c][lexcat][w] = 1
                tokens[author][lexcat] += 1
                if w in types[author][lexcat]:
                    types[author][lexcat][w] += 1
                else:
                    types[author][lexcat][w] = 1


for l in lexcats:
    for c in categories:
        if tokens[c][l] > 0:
            r = len(types[c][l]) / tokens[c][l]
        else:
            r = 0
        print(l, c, r, len(types[c][l].items()), sum(types[c][l].values()), tokens[c][l], r, sep="\t")

