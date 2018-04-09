import sparsevectors
import math
from logger import logger
import json
import pickle

error = True
debug = False
monitor = False


class SemanticSpace:
    def __init__(self, dimensionality=2000, denseness=10):
        self.indexspace = {}
        self.contextspace = {}
        self.associationspace = {}
        self.textspace = {}
        self.utterancespace = {}
        self.authorspace = {}
        self.globalfrequency = {}
        self.bign = 0
        self.dimensionality = dimensionality
        self.denseness = denseness
        self.permutationcollection = {}

    def addoperator(self, item):
        self.permutationcollection[item] = sparsevectors.createpermutation(self.dimensionality)

    def contains(self, item):
        if item in self.indexspace:
            return True
        else:
            return False

    def checkwordspacelist(self, words, loglevel=False):
        for word in words:
            self.checkwordspace(word, loglevel)

    def checkwordspace(self, word, loglevel=False):
        self.bign += 1
        if self.contains(word):
            self.globalfrequency[word] += 1
        else:
            self.additem(word)
            logger(str(word) + " is new and now hallucinated: " + str(self.indexspace[word]), loglevel)

    def observe(self, item):
        self.globalfrequency[item] += 1

    def additem(self, item, vector="dummy"):
        if vector is "dummy":
            vector = sparsevectors.newrandomvector(self.dimensionality, self.denseness)
        if not self.contains(item):
            self.indexspace[item] = vector
            self.globalfrequency[item] = 1
            self.contextspace[item] = sparsevectors.newemptyvector(self.dimensionality)
            self.associationspace[item] = sparsevectors.newemptyvector(self.dimensionality)
#            self.textspace[item] = sparsevectors.newemptyvector(self.dimensionality)
#            self.utterancespace[item] = sparsevectors.newemptyvector(self.dimensionality)
#            self.authorspace[item] = sparsevectors.newemptyvector(self.dimensionality)
            self.bign += 1

    def addsaveditem(self, jsonitem):
        try:
            if self.contains(jsonitem["string"]):
                logger("Conflict in adding new item--- will clobber "+jsonitem["string"], error)
            item = jsonitem["string"]
            self.indexspace[item] = jsonitem["indexvector"]
            self.globalfrequency[item] = int(jsonitem["frequency"])
            self.contextspace[item] = jsonitem["contextvector"]
            self.associationspace[item] = jsonitem["associationvector"]
            self.bign += int(jsonitem["frequency"])
        except:
            logger("Something wrong with item "+jsonitem, error)

    def frequencyweight(self, word):
        try:
            w = math.exp(-300 * math.pi * int(self.globalfrequency[word]) / self.bign)
        except KeyError:
            w = 0.5
        return w

    def outputwordspace(self,filename):
        with open(filename, 'wb') as outfile:
            for item in self.indexspace:
                try:
                    itemj = {}
                    itemj["string"] = str(item)
                    itemj["indexvector"] = self.indexspace[item]
                    itemj["contextvector"] = self.contextspace[item]
                    itemj["associationvector"] = self.associationspace[item]
                    itemj["frequency"] = self.globalfrequency[item]
                    outfile.write(pickle.dumps(itemj, protocol=0))
 #                   outfile.write("\n")
                except TypeError:
                    logger("Could not write >>"+item+"<<", error)

    def importstats(self, wordstatsfile):
        with open(wordstatsfile) as savedstats:
            i = 0
            for line in savedstats:
                i += 1
                try:
                    seqstats = line.rstrip().split("\t")
                    if not self.contains(seqstats[0]):
                        self.additem(seqstats[0])
                    self.globalfrequency[seqstats[0]] = seqstats[1]
                    self.bign += int(seqstats[1])
                except IndexError:
                    logger("***" + str(i) + " " + line.rstrip(), debug)

    def importindexvectors(self, indexvectorfile, frequencythreshold=0):
        cannedindexvectors = open(indexvectorfile, "rb")
        goingalong = True
        n = 0
        m = 0
        while goingalong:
            try:
                itemj = pickle.load(cannedindexvectors)
                item = itemj["string"]
                indexvector = itemj["indexvector"]
                if not self.contains(item):
                    self.additem(item, indexvector)
                    n += 1
                else:
                    if self.globalfrequency[item] > frequencythreshold:
                        self.indexspace[item] = indexvector
                        m += 1
            except EOFError:
                goingalong = False
        return n, m

    def importwordspace(self, wordspacefile, batch=61881):
            i = 0
            logger("Reading weights from " + wordspacefile, monitor)
            with open(wordspacefile) as savedwordspace:
                for line in savedwordspace:
                    i += 1
                    self.addsaveditem(pickle.loads(line))
                    if batch > 0 and i > batch:
                        logger("Skipped rest of weights after " + str(i) + " items.", monitor)
                        break

    def reducewordspace(self, threshold=1):
        items = list(self.indexspace.keys())
        for item in items:
            if self.globalfrequency[item] <= threshold:
                self.removeitem(item)

    def removeitem(self, item):
        if self.contains(item):
            del self.indexspace[item]
            del self.contextspace[item]
            del self.associationspace[item]
            del self.globalfrequency[item]
            self.bign -= 1

    def newemptyvector(self):
        return sparsevectors.newemptyvector(self.dimensionality)

    def similarity(self, item, anotheritem):
        #  should be based on contextspace
        return sparsevectors.sparsecosine(self.indexspace[item], self.indexspace[anotheritem])

