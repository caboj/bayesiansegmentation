import argparse
import numpy as np

parser = argparse.ArgumentParser(description='segment phonemic data')
parser.add_argument('-a', metavar='alpha', dest="alpha",type=float,
                   help='concentration parameter')
parser.add_argument('-tn', metavar='T', dest="trainN",type=int,
                   help='nr of line to train with')


trainD = []       # train data
p0 = {}           # { char : probability of char }
n = 0             # number of utterances in train data
a = 0             # alpha, concentration parameter

def load_data():
    f = open('data/br-phono-train.txt','r')
    global trainD
    global n
    if not n:
        trainD = [ list(i.strip().replace(" ","")) for i in f.readlines()]
        n = len(trainD)
    else:
        trainD = [ list(f.readline().strip().replace(" ","")) for i in range(n)]


    # initialize boundries between words
    boundries = [[] for i in range(n)]
    for i in range(n):
        l = len(trainD[i])
        if l > 2:
            nrBs = int(l/3)+1
            boundries[i] = np.append(np.sort(np.random.choice(np.arange(1,l),replace=False,size=(nrBs))),l)


    define_p0()
    count_words(boundries)
    
def define_p0():
    global p0
    uniqCs = 0
    for sen in trainD:
        for c in sen:
            if c in p0:
                p0[c] += 1
            else:
                p0[c] = 1
                uniqCs += 1

    # a uniform distribution over chars
    for c in p0:
        p0[c] = 1.0/uniqCs

    # a distribution according to relative frequency
    totalCs = sum([len(s) for s in trainD])
    for c in p0:
        p0[c] = float(p0[c])/totalCs


# counts by crp
def count_words(boundries):
    counts = {}
    nwrds = 0
    for i in range(n):
        b0 = 0
        for b in boundries[i]:
            w = ''.join(trainD[i][b0:b])
            nwrds += 1
            if w not in counts:
                counts[w] = [1]
            else:
                if dec_w_novel(nwrds):
                    w_cnts = counts[w]
                    w_cnts.append(1)
                    counts[w] = w_cnts
                else:
                    counts[w][-1] += 1
            b0=b
    for w in counts:
        #if len(counts[w]) > 1:
        print w, counts[w]

def dec_w_novel(nwrds):
    p_novel = a/(nwrds+a)
    p_not_novel = nwrds/(nwrds+a)
    return (p_novel >= p_not_novel)


if __name__ == "__main__":
    args = parser.parse_args()
    n = args.trainN
    a = args.alpha
    load_data()
