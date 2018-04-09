import random
import math
import logger

def sparseadd(onevec, othvec, weight=1, normalised=False):
    if normalised:
        onevec = normalise(onevec)
        othvec = normalise(othvec)
    result = {}
    try:
        for l in onevec:
            result[l] = onevec[l]
        for k in othvec:
            if k in result:
                result[k] = result[k] + othvec[k] * float(weight)
            else:
                result[k] = othvec[k] * float(weight)
    except KeyError:
        logger("error "+str(k)+" "+str(l),error)
        raise
    return result


def sparsemultiply(onevec, othvec, weight=1):
    result = {}
    try:
        for l in onevec:
            if l in othvec:
                result[l] = onevec[l] * othvec[l] * float(weight)
    except KeyError:
        logger("error "+str(k)+" "+str(l),error)
    return result


def sparsexor(onevec, othvec):
    result = {}
    try:
        for l in range(len(onevec)):
            if ((l in onevec) and not (l in othvec)):
                result[l] = 1
            if (not (l in onevec) and (l in othvec)):
                result[l] = 1
    except KeyError:
        logger("error "+str(k)+" "+str(l),error)
    return result

def newrandomvector(n, denseness):
    return {}

def newrandomvector(n, denseness):
    vec = {}
    if denseness % 2 != 0:
        denseness += 1
    if (denseness > 0):  # no need to be careful about this, right? and k % 2 == 0):
        nonzeros = random.sample(list(range(n)), denseness)
        negatives = random.sample(nonzeros, denseness // 2)
        for i in nonzeros:
            vec[str(i)] = 1
        for i in negatives:
            vec[str(i)] = -1
    return vec


def newoperator(n, k):
    return newrandomvector(n, k)


def sparsecosine(xvec, yvec, rounding=True, decimals=4):
    x2 = 0
    y2 = 0
    xy = 0
    try:
        for i in xvec:
            x2 += xvec[i] * xvec[i]
    except KeyError:
        logger("error "+str(i),error)
    try:
        for j in yvec:
            y2 += yvec[j] * yvec[j]
            if j in xvec:
                xy += xvec[j] * yvec[j]
    except KeyError:
        logger("error "+str(j),error)
    if (x2 * y2 == 0):
        cos = 0
    else:
        cos = xy / (math.sqrt(x2) * math.sqrt(y2))
    if (rounding):
        cos = round(cos, decimals)
    return cos


def sparselength(vec, rounding=True):
    x2 = 0
    length = 0
    try:
        for i in vec:
            x2 += vec[i] * vec[i]
    except KeyError:
        logger("error "+str(i),error)
    if (x2 > 0):
        length = math.sqrt(x2)
    if (rounding):
        length = round(length, 4)
    return length


def comb(vec, k, dim):
    newvector = {}
    n = int(k * dim / 2)
    sorted_items = sorted(vec.items(), key=lambda x: x[1])
    bot = sorted_items[:n]
    top = sorted_items[-n:]
    for l in bot:
        newvector[l[0]] = 0
    for l in top:
        newvector[l[0]] = l[1]
    return newvector


def sparsesum(vec):
    s = 0
    for i in vec:
        s += float(vec[i])
    return s


def normalise(vec):
    newvector = {}
    vlen = sparselength(vec, False)
    if (vlen > 0):
        for i in vec:
            newvector[i] = vec[i] / math.sqrt(vlen * vlen)
    else:
        newvector = vec
    return newvector


def modify(vec, factor):
    newvector = {}
    for i in vec:
        if (random.random() > factor):
            newvector[i] = vec[i]
        else:
            newvector[i] = float(vec[i]) * (0.5 - random.random()) * 2.0
    return newvector


def createpermutation(k):
    permutation = random.sample(range(k), k)
    return permutation


def permute(vector, permutation):
    newvector = {}
    try:
        for i in range(len(permutation)):
            if i in vector:  # why was this str(i) for a while? i forget.
                newvector[permutation[i]] = vector[i]
                # newvector[str(permutation[i])] = vector[str(i)]
    except KeyError:
        newvector = vector
        logger("no permutation done, something wrong", error)
    return newvector


def vectorsaturation(vector):
    d = 0
    for c in vector:
        d += 1
    return d


def sparseshift(vector, dimensionality, step=1):
    p = list(range(0, dimensionality))
    return permute(vector, p[step:] + p[:step])
