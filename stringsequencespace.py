import random
import sparsevectors
import pickle
import math

class StringSequenceSpace:
    def __init__(self, dimensionality=2000, denseness=10, window=5):
        self.indexspace = {}
        self.dimensionality = dimensionality
        self.denseness = denseness
        self.window = window
        self.globalfrequency = {}
        self.bign = 0

    def importstats(self, wordstatsfile):
        with open(wordstatsfile) as savedstats:
            i = 0
            for line in savedstats:
                i += 1
                seqstats = line.rstrip().split("\t")
                try:
                    if not seqstats[0] in self.globalfrequency:
                        self.globalfrequency[seqstats[0]] = int(seqstats[1])
                    self.bign += int(seqstats[1])
                except IndexError:
                    pass

    def textvector(self, string, frequencyweighting=True):
        uvector = sparsevectors.newemptyvector(self.dimensionality)
        if self.window > 0:
            windows = [string[ii:ii + self.window] for ii in range(len(string) - self.window + 1)]
            for sequence in windows:
                thisvector = self.makevector(sequence)
                if frequencyweighting:
                    factor = self.frequencyweight(sequence)
                else:
                    factor = 1
                uvector = sparsevectors.sparseadd(uvector, sparsevectors.normalise(thisvector), factor)
        return uvector

    def savecharacterspace(self, filename):
        with open(filename, "wb") as outfile:
            pickle.dump(self.indexspace, outfile)

    def importcharacterspace(self, indexvectorfile):
        self.indexspace = pickle.load(open(indexvectorfile, "rb"))

    def makevector(self, string):
        stringvector = {} #  np.array([0] * self.dimensionality)
        for character in string:
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
