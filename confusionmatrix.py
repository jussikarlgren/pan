class ConfusionMatrix:
    '''ConfusionMatrix keeps book over a categorisation excercise, most notably its errors.'''
    def __init__(self):
        self.confusionmatrix = {}
        self.gold = set()
        self.glitter = set()

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
        macro = 0
        for goldlabel in sorted(self.gold):
            print(goldlabel, "\t", "|", end="")
            weight = 0
            carat = 0
            for glitterlabel in sorted(self.glitter):
                try:
                    if full:
                        print(self.confusionmatrix[goldlabel][glitterlabel], "\t", end="")
                    weight += self.confusionmatrix[goldlabel][glitterlabel]
                    glitterweight[glitterlabel] += self.confusionmatrix[goldlabel][glitterlabel]
                    if glitterlabel == goldlabel:
                        carat = self.confusionmatrix[goldlabel][glitterlabel]
                except KeyError:
                    if full:
                        print(0, "\t", end="")
#            sortedglitter = sorted( self.confusionmatrix[gold].items(),  key=lambda glitter: glitter[1], reverse=True)
            if full:
                print("|", weight, carat, carat / weight, sep="\t")
            hsum += weight
            correct += carat
            if weight > 0:
                macro += carat / weight
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
            macro = macro / len(self.gold)
        micro = correct / hsum
        print("|", hsum, vsum, macro, micro, sep="\t")
        print("\n")
