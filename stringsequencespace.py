import random
import sparsevectors

error = True
debug = False
monitor = False

class StringSequenceSpace:
    def __init__(self, dimensionality=2000, denseness=10, window=5):
        self.indexspace = {}
        self.dimensionality = dimensionality
        self.denseness = denseness
        self.window = window

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

