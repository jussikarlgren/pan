import xml.etree.ElementTree
import nltk
import os
import re
from nltk.tokenize import word_tokenize
import random


from logger import logger

genderfacitfilename = "/home/jussi/data/pan/pan18-author-profiling-training-2018-02-27/en/en.txt"
def readgender(genderfile):
    global facittable
    facittable = {}
    with open(genderfile) as gf:
        for gline in gf:
            authoritem = gline.rstrip().split(":::")
            facittable[authoritem[0]] = authoritem[1]


categories = ["male", "female"]
authornametable = {}

debug = False
monitor = True
error = True

testbatchsize = 1000

resourcedirectory = "/home/jussi/data/pan/pan18-author-profiling-training-2018-02-27/en/text/"
filenamepattern = ".+\.xml"
filenamelist = []
for filename in os.listdir(resourcedirectory):
    hitlist = re.match(filenamepattern, filename)
    if hitlist:  # and random.random() > 0.5:
        filenamelist.append(os.path.join(resourcedirectory, filename))
random.shuffle(filenamelist)
filenamelist = filenamelist[:testbatchsize]

logger("Reading categories from facit file ", monitor)
readgender(genderfacitfilename)

logger("Setting off with a file list of " + str(len(filenamelist)) + " items.", monitor)
stats = {}
freq = {}
lex = {}
pos = {}
patterns = {}

lex["p1pl"] = ["we", "us", "our", "ours", "ourselves"]
pos["p1pl"] = ["PRP"]
lex["p1sg"] = ["i", "me", "my", "mine", "myself"]
pos["p1sg"] = ["PRP"]
lex["p2"] = ["you", "your", "yours", "yourself", "yourselves"]
pos["p2"] = ["PRP"]
pos["think"] = ["VBZ", "VBP", "VBD", "VBN"]
pos["say"] = ["VBZ", "VBP", "VBD", "VBN"]
pos["implicativeverbs"] = ["VBZ", "VBP", "VBD", "VBN"]
lex["think"] = ["think", "believe", "expect", "imagine", "anticipate", "surmise", "suppose", "conjecture", "guess",
              "conclude", "determine", "reason", "reckon", "figure", "opine", "deem", "assess", "judge", "hold",
              "reckon", "consider", "presume", "estimate", "ponder", "reflect", "deliberate", "meditate", "contemplate",
              "muse", "cogitate", "ruminate", "brood", "concentrate", "cerebrate", "consider", "contemplate",
              "deliberate", "mull", "muse", "recall", "remember", "recollect", "imagine", "picture", "visualize",
              "envisage", "envision", "dream", "fantasize", "suppose", "assume", "imagine", "presume", "hypothesis",
              "hypothesize", "postulate", "posit", "evaluate", "judge", "gauge", "rate", "estimate", "appraise",
              "analyse", "worry", "fret", "conceive"]
lex["say"] = ["say", "speak", "utter", "voice", "pronounce", "vocalize", "declare", "state", "announce", "remark",
            "observe", "mention", "comment", "note", "add", "reply", "respond", "answer", "rejoin", "whisper", "mutter",
            "mumble", "mouth", "claim", "maintain", "assert", "hold", "insist", "contend", "aver", "affirm", "avow",
            "allege", "profess", "opine", "asseverate", "express", "phrase", "articulate", "communicate", "convey",
            "verbalize", "render", "tell", "reveal", "divulge", "impart", "disclose", "imply", "suggest", "signify",
            "denote", "mean", "recite", "repeat", "utter", "deliver", "perform", "declaim", "orate", "indicate",
            "estimate", "judge", "guess", "hazard a guess", "dare say", "predict", "speculate", "surmise", "conjecture",
            "venture", "pontificate", "propose", "plead", "mutter", "murmur", "mumble", "whisper", "hush", "grumble",
            "moan", "complain", "grouse", "carp", "whine", "bleat", "gripe", "whinge", "whine", "kvetch", "cry", "yelp",
            "call", "shout", "howl", "yowl", "wail", "scream", "shriek", "screech", "squawk", "squeal", "roar", "bawl",
            "whoop", "holler", "ululate", "laugh", "chuckle", "chortle", "guffaw", "giggle", "titter", "snigger",
            "snicker", "cackle", "howl", "roar", "smile", "ridicule", "mock", "deride", "scoff", "jeer", "sneer",
            "jibe", "scorn", "lampoon", "satirize", "caricature", "parody", "taunt", "tease", "torment", "expound",
            "declaim", "preach", "express", "sermonize", "moralize", "pronounce", "lecture", "expatiate", "spiel",
            "perorate"]
lex["implicativeverbs"] = ["manage", "forget", "fail", "obey", "succeed"]

amplifyGrade = ["very", "awfully", "completely", "enormously", "entirely", "exceedingly", "excessively", "extremely",
                "greatly", "highly", "hugely", "immensely", "intensely", "particularly", "radically", "significantly",
                "strongly", "substantially", "totally", "utterly", "vastly"]

amplifyTruly = ["absolutely", "definitely", "famously", "genuinely", "immaculately", "overly", "perfectly", "really",
                "severely", "surely", "thoroughly", "truly", "undoubtedly"]

amplifySurprise = ["amazingly", "dramatically", "drastically", "emphatically", "exceptionally", "extraordinarily",
                   "fantastically", "horribly", "incredibly", "insanely", "phenomenally", "remarkably", "ridiculously",
                   "strikingly", "surprisingly", "terribly", "unusually", "wildly", "wonderfully"]
pos["neg"] = ["RB", "CC", "DT"]
lex["neg"] = ["no", "none", "never", "not", "n't", "neither", "nor"]
lex["amp"]  = amplifyGrade + amplifySurprise + amplifyTruly
pos["amp"] = ["RB", "VB"]
lex["ampG"] = amplifyGrade
lex["ampT"] = amplifyTruly
lex["ampS"] = amplifySurprise
pos["ampT"] = pos["ampS"] = pos["ampG"] = pos["amp"]
pos["hedge"] = pos["amp"] + pos["say"] + ["JJ"]
lex["herenow"] = ["here", "now", "this", "these", "that", "those"]
pos["herenow"] = ["RB", "DT"]
lex["hedge"] = ["apparently", "appear", "around", "basically", "effectively", "evidently", "fairly", "generally",
             "hopefully", "largely", "likely", "mainly", "maybe", "mostly", "overall", "perhaps", "presumably",
             "probably", "quite", "rather", "somewhat", "supposedly", "possibly", "doubtfully", "arguably", "often",
             "unlikely", "usually", "sometimes", "certainly", "definitely", "clearly", "conceivably", "apparent",
             "certain", "possible", "presumed", "probable", "putative", "supposed", "doubtful", "appear", "assume",
             "estimate", "indicate", "infer", "intend", "presume", "propose", "seem", "speculate", "suggest", "suppose",
             "tend", "doubt"]

patterns["hashtag"] = r"\#\w+"
patterns["allcaps"] = r"\W[A-Z]+\W"
patterns["initcaps"] = r"\W[A-Z]\w+"
patterns["multipunct"] = r'[\.\:\?\!\,\;]{2,}'

seen = {}
bign = 0
for c in categories:
    stats[c] = {}
    freq[c] = 0
    for f in lex:
        stats[c][f] = 0
    for f in patterns:
        stats[c][f] = 0
for f in lex:
    seen[f] = 0
for f in patterns:
    seen[f] = 0

for file in filenamelist:
    author = file.split(".")[0].split("/")[-1]
    e = xml.etree.ElementTree.parse(file).getroot()
    for b in e.iter("document"):
        bign += 1
        text = word_tokenize(b.text)
        poses = nltk.pos_tag(text)
        freq[facittable[author]] += 1
        for f in lex:
            if any([x for x in poses if x[0] in lex[f] and x[1] in pos[f]]):
                stats[facittable[author]][f] += 1   # occurs or not, not frequency!
                seen[f] += 1
        for f in patterns:
            c = len(re.findall(patterns[f], b.text))
            stats[facittable[author]][f] += 1
            seen[f] += 1

for f in seen:
    k = 0
    print("====")
    print(" ", f, end=" |\t")
    for c in categories:
        print(c, end="\t")
    print(" |")
    print("---------" * (2 + len(categories)))
    print("\t 1 |", end="\t")
    for c in categories:
        try:
            print(str(stats[c][f]), end="\t\t")
        except KeyError:
            print("0", end="\t\t")
    print(" |", str(seen[f]), sep="\t")
    print("\t e |", end="\t")
    for c in categories:
        e = seen[f] * freq[c] / bign
        try:
            k += (stats[c][f] - e) ** 2 / stats[c][f]
        except ZeroDivisionError:
            pass
        try:
            k += (freq[c] - stats[c][f] - (bign - seen[f]) * freq[c] / bign) ** 2 / (freq[c] - stats[c][f])
        except ZeroDivisionError:
            pass
        try:
            print(str(int(e)), end="\t\t")
        except KeyError:
            print("0", end="\t\t")
    print(" |", str(seen[f]), sep="\t")
    print("\t 0 |", end="\t")
    for c in categories:
        try:
            print(str(freq[c] - stats[c][f]), end="\t\t")
        except KeyError:
            print(str(freq[c]), end="\t\t")
    print(" |", str(bign - seen[f]), sep="\t")
    print("---------" * (2 + len(categories)))
    print("\t t |", end="\t")
    for c in categories:
        print(str(freq[c]), end="\t\t")
    print(" |", str(bign))
    print("\t\t\t\t\t\t\t",k)

