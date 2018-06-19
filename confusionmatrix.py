class ConfusionMatrix:
    '''ConfusionMatrix keeps book over a categorisation excercise, most notably its errors.'''
    def __init__(self):
        self.confusionmatrix = {}
        self.gold = set()         # actual labels
        self.glitter = set()      # predicted labels
        self.weight = {}          # number of actual items for a category
        self.glitterweight = {}   # number of items with a category label
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
        for glitterlabel in sorted(self.glitter):
            print(glitterlabel, end="\t")
            self.glitterweight[glitterlabel] = 0
        print("|", "sum", "correct", "recall", sep="\t")
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
                    self.glitterweight[glitterlabel] += self.confusionmatrix[goldlabel][glitterlabel]
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
            vsum += self.glitterweight[glitterlabel]
            print(self.glitterweight[glitterlabel], "\t", end="")
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
        # PRECISION LINE
        print("precision", "|", sep="\t", end="\t")
        for glitterlabel in sorted(self.glitter):
            if self.glitterweight[glitterlabel] > 0:
                try:
                    print(self.carat[glitterlabel] / self.glitterweight[glitterlabel], "\t", end="")
                except KeyError:
                    print("0.0", "\t", end="")
            else:
                print("0.00")
        print("|")

        return self.macro
