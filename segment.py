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
cur_tot_words = 0 # word count
counts = {}

def load_data():
    f = open('data/br-phono-train.txt','r')
    global trainD
    global n
    if not n:
        trainD = [ list(i.strip().replace(" ","")) for i in f.readlines()]
        n = len(trainD)
    else:
        trainD = [ list(f.readline().strip().replace(" ","")) for i in range(n)]

    n_uf = n
    # initialize boundries between words
    boundaries = [[] for i in range(n)]
    for i in range(n):
        l = len(trainD[i])
        if l > 2:
            nrBs = int(l/3)+1
            boundaries[i] = np.append(np.sort(np.random.choice(np.arange(1,l),replace=False,size=(nrBs))),l)


    define_p0()
    count_words(boundaries)

    return boundaries
    
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
def count_words(boundaries):
    global counts
    counts = {}
    nwrds = 0
    for i in range(n):
        b0 = 0
        for b in boundaries[i]:
            w = ''.join(trainD[i][b0:b])
            nwrds += 1
            if w not in counts:
                counts[w] = 1
            else:
                counts[w] += 1
                '''
                if dec_w_novel(nwrds):
                    w_cnts = counts[w]
                    w_cnts.append(1)
                    counts[w] = w_cnts
                else:
                    counts[w][-1] += 1
                '''
            b0=b
    global cur_tot_words
    cur_tot_words = nwrds

'''
def dec_w_novel(nwrds):
    p_novel = a/(nwrds+a)
    p_not_novel = nwrds/(nwrds+a)
    return (p_novel >= p_not_novel)
'''

def prob_h1(word):
    if word in counts:
        nw1 = counts[word] - 1
    else:
        nw1 = 0

    p0_w = 1
    for c in list(word):
        p0_w *= p0[c]

    return ((nw1+ a*p0_w)/(cur_tot_words -1 + a))  


def prob_h2(word1,word2):
    if word1 in counts:
        nw1 = counts[word1] - 1
    else:
        nw1 = 0
    if word2 in counts:
        nw2 = counts[word2] - 1
    else:
        nw2 = 0

    p0_w1 = 1
    for c in list(word1):
        p0_w1 *= p0[c]

    p0_w2 = 1
    for c in list(word1):
        p0_w2 *= p0[c]

        
    return ((nw1+ a*p0_w1)/(cur_tot_words -1 + a))* ((nw2+ a*p0_w2)/(cur_tot_words -1 + a))


def gibbs(boundaries):
    for ut,bndrs in zip(trainD,boundaries):
        b0 = 0
        for b in bndrs:
            w1 = ''.join([ut[c] for c in range(b0,b)])
            p_h1 = prob_h1(w1)
            for i in range(b0,b):
                w2 = ''.join([ut[c] for c in range(b0,i)])
                w3 = ''.join([ut[c] for c in range(i,b)])
                p_h2 = prob_h2(w2,w3)
                if p_h2 > p_h1:
                    print(' new boundary ')

    
if __name__ == "__main__":
    args = parser.parse_args()
    n = args.trainN
    a = args.alpha
    gibbs(load_data())
    
