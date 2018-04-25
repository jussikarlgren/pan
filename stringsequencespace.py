import random
import sparsevectors
import pickle
import math
from logger import logger
import nltk

error = True

class StringSequenceSpace:
    def __init__(self, dimensionality=2000, denseness=10, window=5):
        self.indexspace = {}
        self.dimensionality = dimensionality
        self.denseness = denseness
        self.window = window
        self.globalfrequency = {}
        self.bign = 0

    def dummy(self):
        print("dummy")

    def importstats(self, wordstatsfile, loglevel=False):
        logger(wordstatsfile, loglevel)
        with open(wordstatsfile, "r") as savedstats:
            i = 0
            try:
                for line in savedstats:
                    i += 1
                    seqstats = line.rstrip().split("\t")
                    try:
                        if not seqstats[0] in self.globalfrequency:
                            self.globalfrequency[seqstats[0]] = int(seqstats[1])
                        self.bign += int(seqstats[1])
                    except IndexError:
                        pass
            except UnicodeDecodeError:
                logger("Error at line " + str(i), error)

    def textvector(self, string, frequencyweighting=True, loglevel=False):
        logger("----" + string, loglevel)
        uvector = sparsevectors.newemptyvector(self.dimensionality)
        if self.window > 0:
            windows = [string[ii:ii + self.window] for ii in range(len(string) - self.window + 1)]
            for sequence in windows:
                thisvector = self.makevector(sequence)
                if frequencyweighting:
                    factor = self.frequencyweight(sequence)
                else:
                    factor = 1
                logger(sequence + " " + str(factor), loglevel)
                if loglevel:
                    logger(str(sparsevectors.sparsecosine(uvector, sparsevectors.normalise(thisvector))), loglevel)
                uvector = sparsevectors.sparseadd(uvector, sparsevectors.normalise(thisvector), factor)
        else:
            words = nltk.word_tokenize(string)
            for w in words:
                if frequencyweighting:
                    factor = self.frequencyweight(w)
                else:
                    factor = 1
                if w not in self.indexspace:
                    self.indexspace[w] = sparsevectors.newrandomvector(self.dimensionality, self.denseness)
                uvector = sparsevectors.sparseadd(uvector, sparsevectors.normalise(self.indexspace[w]), factor)
        return uvector

    def savecharacterspace(self, filename):
        with open(filename, "wb") as outfile:
            pickle.dump(self.indexspace, outfile)

    def importcharacterspace(self, indexvectorfile):
        self.indexspace = pickle.load(open(indexvectorfile, "rb"))

    def makevector(self, string):
        stringvector = {}  #  np.array([0] * self.dimensionality)
        for character in string[::-1]:    # reverse the string! (to keep strings that share prefixes similar)
            if character not in self.indexspace:
                vec = {}
                nonzeros = random.sample(list(range(self.dimensionality)), self.denseness)
                random.shuffle(nonzeros)
                split = self.denseness // 2
                for i in nonzeros[:split]:
                    vec[i] = 1
                for i in nonzeros[split:]:
                    vec[i] = -1
                self.indexspace[character] = vec
            stringvector = sparsevectors.sparseadd(sparsevectors.sparseshift(stringvector, self.dimensionality),
                                                   self.indexspace[character])
            # np.append(stringvector[1:], stringvector[0]) + self.indexspace[character]
        return stringvector  # lil_matrix(stringvector.reshape(self.dimensionality, -1))

    def frequencyweight(self, word):
        try:
            w = math.exp(-300 * math.pi * int(self.globalfrequency[word]) / self.bign)
        except KeyError:
            w = 0.5
        return w

