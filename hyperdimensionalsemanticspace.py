import sparsevectors
import math
from logger import logger
import json

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


    def addoperator(self,item):
        self.permutationcollection[item] = sparsevectors.createpermutation(self.dimensionality)

    def contains(self, item):
        if item in self.indexspace:
            return True
        else:
            return False

    def checkwordspacelist(self, words, debug=False):
        for word in words:
            self.checkwordspace(word,debug)

    def checkwordspace(self, word, debug=False):
        self.bign += 1
        if self.contains(word):
            self.globalfrequency[word] += 1
        else:
            self.additem(word)
            logger(str(word) + " is new and now hallucinated.", debug)

    def observe(self,item):
        self.globalfrequency[item] += 1

    def additem(self, item, vector=None):
        if not self.contains(item):
            if vector:
                self.indexspace[item] = vector
            else:
                self.indexspace[item] = sparsevectors.newrandomvector(self.dimensionality, self.denseness)
            self.globalfrequency[item] = 1
            self.contextspace[item] = {}
            self.associationspace[item] = {}
            self.textspace[item] = {}
            self.utterancespace[item] = {}
            self.authorspace[item] = {}
            self.bign += 1

    def addsaveditem(self, jsonitem):
        try:
            if self.contains(jsonitem["string"]):
                logger("Conflict in adding new item--- will clobber "+string, error)
            item = jsonitem["string"]
            self.indexspace[item] = jsonitem["indexvector"]
            self.globalfrequency[item] = jsonitem["frequency"]
            self.contextspace[item] = jsonitem["contextvector"]
            self.associationspace[item] = jsonitem["associationvector"]
            self.bign += int(jsonitem["frequency"])
        except:
            logger("Something wrong with item "+jsonitem, error)

    def frequencyweight(self, word):
        try:
            w = math.exp(-300 * math.pi * self.globalfrequency[word] / self.bign)
        except KeyError:
            w = 0.5
        return w


    def outputwordspace(self,filename):
        with open(filename, 'w') as outfile:
            for item in self.indexspace:
                try:
                    itemj = {}
                    itemj["string"] = str(item)
                    itemj["indexvector"] = self.indexspace[item]
                    itemj["contextvector"] = self.contextspace[item]
                    itemj["associationvector"] = self.associationspace[item]
                    itemj["frequency"] = self.globalfrequency[item]
                    outfile.write(json.dumps(itemj)+"\n")
                except TypeError:
                    logger("Could not write >>"+item+"<<", error)


