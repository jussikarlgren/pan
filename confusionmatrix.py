class ConfusionMatrix:
    '''ConfusionMatrix keeps book over a categorisation excercise, most notably its errors.'''
    def __init__(self):
        self.confusionmatrix = {}
        self.gold = set()
        self.glitter = set()
        self.weight = {}
        self.carat = {}
        self.macro = 0
        self.micro = 0

    def addconfusion(self, rfacit, rpredicted):
        facit = str(rfacit)
        self.gold.add(facit)
        predicted = str(rpredicted)
        self.glitter.add(predicted)
        if facit in self.confusionmatrix:
            if predicted in self.confusionmatrix[facit]:
                self.confusionmatrix[facit][predicted] += 1
            else:
                self.confusionmatrix[facit][predicted] = 1
        else:
            self.confusionmatrix[facit] = {}
            self.confusionmatrix[facit][predicted] = 1

    def evaluate(self, full=True):
        print("\t|", end="")
        glitterweight = {}
        for glitterlabel in sorted(self.glitter):
            print(glitterlabel, "\t", end="")
            glitterweight[glitterlabel] = 0
        print("|\tsum\n")
        hsum = 0
        correct = 0
        for goldlabel in sorted(self.gold):
            print(goldlabel, "\t", "|", end="")
            self.weight[goldlabel] = 0
            self.carat[goldlabel] = 0
            for glitterlabel in sorted(self.glitter):
                try:
                    if full:
                        print(self.confusionmatrix[goldlabel][glitterlabel], "\t", end="")
                        self.weight[goldlabel] += self.confusionmatrix[goldlabel][glitterlabel]
                    glitterweight[glitterlabel] += self.confusionmatrix[goldlabel][glitterlabel]
                    if glitterlabel == goldlabel:
                        self.carat[goldlabel] = self.confusionmatrix[goldlabel][glitterlabel]
                except KeyError:
                    if full:
                        print(0, "\t", end="")
#            sortedglitter = sorted( self.confusionmatrix[gold].items(),  key=lambda glitter: glitter[1], reverse=True)
            if full:
                print("|", self.weight[goldlabel], self.carat[goldlabel],
                      self.carat[goldlabel] / self.weight[goldlabel], sep="\t")
            hsum += self.weight[goldlabel]
            correct += self.carat[goldlabel]
            if self.weight[goldlabel] > 0:
                self.macro += self.carat[goldlabel] / self.weight[goldlabel]
        if full:
            for glitterlabel in sorted(self.glitter):
                print("--------", end="\t")
            print("--------")
            print("\t|",end="")
        vsum = 0
        for glitterlabel in sorted(self.glitter):
            vsum += glitterweight[glitterlabel]
            if full:
                print(glitterweight[glitterlabel], "\t", end="")
        if hsum > 0:
            self.macro = self.macro / len(self.gold)
            self.micro = correct / hsum
        print("|", hsum, vsum, self.macro, self.micro, sep="\t")
        print("\n")
        return self.macro
