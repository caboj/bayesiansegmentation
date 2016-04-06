import argparse
import numpy as np

parser = argparse.ArgumentParser(description='segment phonemic data')
parser.add_argument('-tn', metavar='T', dest="trainN",type=int,
                   help='nr of line to train with')


trainD = []       # train data
p0 = {}           # { char : probability of char }
n = 0             # number of utterances in train data

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


# tokens per type
def count_words(boundries):
    counts = {}
    for i in range(n):
        b0 = 0
        for b in boundries[i]:
            w = ''.join(trainD[i][b0:b])
            if w in counts:
                counts[w] += 1
            else:
                counts[w] = 1
            b0=b


if __name__ == "__main__":
    args = parser.parse_args()
    n = args.trainN
    load_data()
