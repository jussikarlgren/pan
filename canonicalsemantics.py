import hyperdimensionalsemanticspace
import tweetfilereader
from logger import logger
import stringsequencespace
from nltk import sent_tokenize
from nltk import word_tokenize
import os

error = True
monitor = True
debug = False

pairs = {}
vectors = {}

canonicalwordset = set()
attributewordset = set()
amplifierGwordset = set()
amplifierSwordset = set()
amplifierTwordset = set()
pragmaticswordset = set()
negattitudewordset = set()
posattitudewordset = set()
downtonerswordset = set()
negationwordset = set()
dim = 2000
den = 10
win = 0
space = hyperdimensionalsemanticspace.SemanticSpace(dim, den)
strings = stringsequencespace.StringSequenceSpace(dim, den, win)
testbatchsize = 10000
batch = 500
thresholdofinterest = 5
negationskipwindow = 4
amplifierdowntonerwindow = 4
datadirectory= "/home/jussi/data/storm/fixed/"
outputdirectory="/home/jussi/data/wordspaces/"
resourcedirectory="/home/jussi/data/poles/en/"

def redovisa(n, file="canonical.space"):
    print(n)
    for cw in canonicalwordset:
        try:
            if space.globalfrequency[cw] > thresholdofinterest:
                print(cw, space.globalfrequency[cw], sep="\t")
#                      antalnegationer[cw],
#                      antaldt[cw], antalampG[cw], antalampS[cw], antalampT[cw], sep="\t")
#                ns = space.contextneighbourswithweights(cw, n)
#                for item in ns:
##                    if item[0] == cw:
#                        continue
#                    print(item, sep="\t", end=" ")
#                    print()
        except KeyError:
            logger("***" + cw, error)
    space.outputwordspace(os.path.join(datadirectory, file))


with open("/home/jussi/data/poles/en/enposBingLiu.list", "r") as posfile:
    space.addconstant("JiKpositive")
    line = posfile.readline()
    lineno = 0
    while line:
        lineno += 1
        word = line.rstrip()
        posattitudewordset.add(word)
        space.additem(word)
        line = posfile.readline()

with open("/home/jussi/data/poles/en/ennegBingLiu.list", "r") as negfile:
    line = negfile.readline()
    lineno = 0
    space.addconstant("JiKnegative")
    while line:
        lineno += 1
        word = line.rstrip()
        negattitudewordset.add(word)
        space.additem(word)
        line = negfile.readline()

with open("/home/jussi/data/poles/en/enpurenegationnltk.list", "r") as negfile:
    space.addconstant("JiKnegation")
    line = negfile.readline()
    lineno = 0
    while line:
        lineno += 1
        word = line.rstrip()
        negationwordset.add(word)
        space.additem(word)
        line = negfile.readline()

with open("/home/jussi/data/poles/en/aspectualattributesofinterest.txt", "r") as attributefile:
    line = attributefile.readline()
    lineno = 0
    space.addconstant("JiKtensemoodaspect")
    while line:
        lineno += 1
        word = line.rstrip()
        attributewordset.add(word)
        space.additem(word)
        line = attributefile.readline()

with open("/home/jussi/data/poles/en/pragmaticsituationmarkers.txt", "r") as pragmaticsfile:
    line = pragmaticsfile.readline()
    lineno = 0
    space.addconstant("JiKhereandnow")
    while line:
        lineno += 1
        word = line.rstrip()
        pragmaticswordset.add(word)
        space.additem(word)
        line = pragmaticsfile.readline()

with open("/home/jussi/data/poles/en/downtoners.txt", "r") as dtfile:
    line = dtfile.readline()
    lineno = 0
    space.addconstant("JiKhedge")
    while line:
        lineno += 1
        word = line.rstrip()
        downtonerswordset.add(word)
        space.additem(word)
        line = dtfile.readline()

with open("/home/jussi/data/poles/en/enamplifyGrade.list", "r") as ampfile:
    line = ampfile.readline()
    lineno = 0
    space.addconstant("JiKampgrade")
    while line:
        lineno += 1
        word = line.rstrip()
        amplifierGwordset.add(word)
        space.additem(word)
        line = ampfile.readline()

with open("/home/jussi/data/poles/en/enamplifySurprise.list", "r") as ampfile:
    line = ampfile.readline()
    lineno = 0
    space.addconstant("JiKampsurprise")
    while line:
        lineno += 1
        word = line.rstrip()
        amplifierSwordset.add(word)
        space.additem(word)
        line = ampfile.readline()

with open("/home/jussi/data/poles/en/enamplifyTruly.list", "r") as ampfile:
    line = ampfile.readline()
    lineno = 0
    space.addconstant("JiKamptruly")
    while line:
        lineno += 1
        word = line.rstrip()
        amplifierTwordset.add(word)
        space.additem(word)
        line = ampfile.readline()


# with open("/home/jussi/data/poles/en/canonicalpairs.txt", "r") as canonicalpairsfile:
#     line = canonicalpairsfile.readline()
#     lineno = 0
#     while line:
#         lineno += 1
#         try:
#             one, other = line.rstrip().split("\t")
#             pairs[one] = other
#             pairs[other] = one
#             space.additem(one)
#             space.additem(other)
#             canonicalwordset.add(one)
#             canonicalwordset.add(other)
#         except ValueError:
#             logger("***" + line, error)
#         line = canonicalpairsfile.readline()

with open("/home/jussi/data/poles/en/canonicalwords.txt", "r") as canonicalwordsfile:
    line = canonicalwordsfile.readline()
    lineno = 0
    while line:
        lineno += 1
        one = line.rstrip()
        space.additem(one)
        canonicalwordset.add(one)
        line = canonicalwordsfile.readline()


def getsentencesfromlinefile(filename="/home/jussi/data/newsprint/ap10k.txt"):
    sentences = []
    with open(filename, "r") as newsfile:
        newsline = newsfile.readline()
        nl = 0
        while newsline:
            nl += 1
            if nl > testbatchsize:
                break
            author, text = newsline.split("\t")
            sents = sent_tokenize(text)
            sentences = sentences + sents
            newsline = newsfile.readline()
    return sentences


def countitemsinsentences(sents):
    global antals, antaldt, antaltreff, antalnegationer, antalampS, antalampG, antalampT
    for s in sents:
        logger(str(s), debug)
        antals += 1
        words = word_tokenize(s)
        targetwords = canonicalwordset.intersection(words)
        if targetwords:
            antaltreff += 1
            vec = strings.textvector(s)
            negflag = 0
            ampSflag = 0
            ampTflag = 0
            ampGflag = 0
            dtflag = 0
            for nn in words:
                if nn in negationwordset:
                    negflag = negationskipwindow
                if nn in amplifierGwordset:
                    ampGflag = amplifierdowntonerwindow
                if nn in amplifierTwordset:
                    ampTflag = amplifierdowntonerwindow
                if nn in amplifierSwordset:
                    ampSflag = amplifierdowntonerwindow
                if nn in downtonerswordset:
                    dtflag = amplifierdowntonerwindow
                if nn in targetwords:
                    if negflag > 0:
                        negflag = 0
                        space.observecollocation(nn, "JiKnegation")
                        antalnegationer[nn] += 1
                    if dtflag > 0:
                        dtflag = 0
                        space.observecollocation(nn, "JiKhedge")
                        antaldt[nn] += 1
                    if ampGflag > 0:
                        ampGflag = 0
                        space.observecollocation(nn, "JiKampgrade")
                        antalampG[nn] += 1
                    if ampSflag > 0:
                        ampSflag = 0
                        space.observecollocation(nn, "JiKampsurprise")
                        antalampS[nn] += 1
                    if ampTflag > 0:
                        ampTflag = 0
                        space.observecollocation(nn, "JiKamptruly")
                        antalampT[nn] += 1
                negflag -= 1
                ampSflag -= 1
                ampTflag -= 1
                ampGflag -= 1
                dtflag -= 1

            for w in targetwords:
                space.addintoitem(w, vec)
                space.observe(w)
                if debug:
                    print(w, space.globalfrequency[w], sep="\t", end="\t")
                    ns = space.contextneighbours(w)
                    for n in ns:
                        if n == w:
                            continue
                        print(n, sep="\t", end=" ")
                    print()
            pragmaticwords = pragmaticswordset.intersection(words)
            for p in pragmaticwords:
                for w in targetwords:
                    space.observecollocation(w, "JiKherenow")
            aspectwords = attributewordset.intersection(words)
            for a in aspectwords:
                for w in targetwords:
                    space.observecollocation(w, "JiKtensemoodaspect")
            poswords = posattitudewordset.intersection(words)
            for a in poswords:
                for w in targetwords:
                    if w == a:
                        continue
                    space.observecollocation(w, "JiKpositive")
            negwords = negattitudewordset.intersection(words)
            for a in negwords:
                for w in targetwords:
                    if w == a:
                        continue
                    space.observecollocation(w, "JiKnegative")


antals = 0
antaltreff = 0
antalnegationer = {}
antalampG = {}
antalampS = {}
antalampT = {}
antaldt = {}
for w in canonicalwordset:
    antalnegationer[w] = 0
    antalampG[w] = 0
    antalampS[w] = 0
    antalampT[w] = 0
    antaldt[w] = 0
fs = tweetfilereader.getfilelist()
nn = len(fs)
n0 = 0
for f in fs:
    n0 += 1
    logger(str(n0) + f, debug)
    sents = tweetfilereader.dotweetfiles(datadirectory, [f], debug)
    countitemsinsentences(sents)
    redovisa(n0, "/home/jussi/data/wordspaces/canonical.space." + f)