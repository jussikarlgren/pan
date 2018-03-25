class ConfusionMatrix:
    '''ConfusionMatrix keeps book over a categorisation excercise, most notably its errors.'''
    def __init__(self):
        self.confusionmatrix = {}

    def addconfusion(self, rfacit, rpredicted):
        facit = str(rfacit)
        predicted = str(rpredicted)
        if facit in self.confusionmatrix:
            if predicted in self.confusionmatrix[facit]:
                self.confusionmatrix[facit][predicted] += 1
            else:
                self.confusionmatrix[facit][predicted] = 1
        else:
            self.confusionmatrix[facit] = {}
            self.confusionmatrix[facit][predicted] = 1

    def evaluate(self):
        for gold in sorted(self.confusionmatrix):
            print("---")
            carat = 0
            maximum = 0
            hitn = 0
            sortedglitter = sorted(
                self.confusionmatrix[gold].items(),
                key=lambda glitter: glitter[1],
                reverse=True)
            for glitter in sortedglitter:
                hit = ""
                carat += glitter[1]
                if glitter[0] == gold:
                    hit = "***"
                    hitn = glitter[1]
                print(gold, glitter[0], glitter[1], hit, sep="\t")
            print(gold, "sum", hitn, carat, hitn / carat, sep="\t")

