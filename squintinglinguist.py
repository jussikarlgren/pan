from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag


def generalise(text):
    accumulator = []
    sents = sent_tokenize(text)
    for sentence in sents:
        words = word_tokenize(sentence)
        poses = pos_tag(words)
        for item in poses:
            if item[1] == "NN":
                accumulator.append("N")
            else:
                accumulator.append(item[0])
    return " ".join(accumulator)
