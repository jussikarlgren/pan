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
        # HEADER LINE
        print("\t\t|\t", end="")
        glitterweight = {}
        for glitterlabel in sorted(self.glitter):
            print(glitterlabel, end="\t")
            glitterweight[glitterlabel] = 0
        print("|", "sum", "correct", "precision", sep="\t")
        # SEPARATOR LINE
        for gg in range(2 + len(self.glitter)):
            print("--------", end="\t")
        print()

        hsum = 0
        correct = 0
        for goldlabel in sorted(self.gold):
            print(goldlabel, "|", sep="\t", end="\t")
            self.weight[goldlabel] = 0
            self.carat[goldlabel] = 0
            for glitterlabel in sorted(self.glitter):
                try:
                    print(self.confusionmatrix[goldlabel][glitterlabel], "\t", end="")
                    self.weight[goldlabel] += self.confusionmatrix[goldlabel][glitterlabel]
                    glitterweight[glitterlabel] += self.confusionmatrix[goldlabel][glitterlabel]
                    if glitterlabel == goldlabel:
                        self.carat[goldlabel] = self.confusionmatrix[goldlabel][glitterlabel]
                except KeyError:
                    print(0, "\t", end="")
            print("|", self.weight[goldlabel], self.carat[goldlabel],
                  self.carat[goldlabel] / self.weight[goldlabel], sep="\t")
            hsum += self.weight[goldlabel]
            correct += self.carat[goldlabel]
            if self.weight[goldlabel] > 0:
                self.macro += self.carat[goldlabel] / self.weight[goldlabel]
        # SEPARATOR LINE
        for gg in range(2 + len(self.glitter)):
            print("--------", end="\t")
        print()

        # SUM of PREDICTIONS LINE
        print("sum", "|", sep="\t\t", end="\t")
        vsum = 0
        for glitterlabel in sorted(self.glitter):
            vsum += glitterweight[glitterlabel]
            print(glitterweight[glitterlabel], "\t", end="")
        if hsum > 0:
            self.macro = self.macro / len(self.gold)
            self.micro = correct / hsum
        print("|", hsum, vsum, self.macro, self.micro, sep="\t")
        # CORRECT LINE
        print("correct", "|", sep="\t", end="\t")
        for glitterlabel in sorted(self.glitter):
            try:
                print(self.carat[glitterlabel], "\t", end="")
            except KeyError:
                print("0.0", "\t", end="")
        print("|")
        # RECALL LINE
        print("recall", "|", sep="\t", end="\t")
        for glitterlabel in sorted(self.glitter):
            if glitterweight[glitterlabel] > 0:
                try:
                    print(self.carat[glitterlabel] / self.weight[glitterlabel], "\t", end="")
                except KeyError:
                    print("0.0", "\t", end="")
            else:
                print("0.00")
        print("|")

        return self.macro
