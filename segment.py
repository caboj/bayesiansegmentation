import argparse
import numpy as np
from collections import Counter, defaultdict

parser = argparse.ArgumentParser(description='segment phonemic data')
parser.add_argument('-a', metavar='alpha', dest="alpha",type=float,
                   help='concentration parameter')
parser.add_argument('-tn', metavar='T', dest="trainN",type=int,
                   help='nr of line to train with')
parser.add_argument('-ep', metavar='epochs', dest="epochs",type=int,
                   help='nr of sampling epochs', default=1)
parser.add_argument('-rho', metavar='rho', dest="rho",type=float,
                   help='', default=0.5)
parser.add_argument('-s', metavar='save', dest="save",type=bool,
                   help='', default=True)

save = True                 # save result
rho = 0                     # rho
epochs = 1                  # epochs
data = []                   # read data
trainD = []                 # train data
trainC = np.array([])       # encoded train data
p0 = np.array([])           # { char : probability of char }
voc = []                    # vocabulary
n = 0                       # number of utterances in train data
a = 0                       # alpha, concentration parameter
cur_tot_words = 0           # nr of words
counts = defaultdict(int)   # word counts

def load_data():
    global data
    global trainD
    global n

    with open('data/br-phono-train.txt', 'rU') as f:
        if not n:
            data = np.array([np.array( (i.strip()).split(' ') ) for i in f.readlines()])
            n = len(trainD)
        else:
            data = np.array([np.array( (f.readline().strip()).split(' ') ) for i in range(n)])
    trainD = np.array([np.array(list((' '.join(s)).replace(" ", ""))) for s in data])

    n_uf = n
    # initialize boundries between words
    bounds = [[] for i in range(n)]
    for i in range(n):
        l = len(trainD[i])
        if l > 2:
            nrBs = int(l/3)+1
            bounds[i] = np.append(np.sort(np.random.choice(np.arange(1,l),replace=False,size=(nrBs))),l)

    define_voc()
    define_trainC()
    define_p0()
    count_words(bounds)
    return bounds

def define_p0():
    global p0
    p0 = np.zeros(len(voc))
    # a uniform distribution over chars
    p0[:] = 1.0/len(voc)
    # a distribution according to relative frequency
    totalCs = len(np.hstack(trainC))
    p0 = p0[:]/float(totalCs)

def define_trainC():
    global trainC
    trainC = np.array([np.array([voc.index(c) for c in s]) for s in trainD])

def define_voc():
    global voc
    flat = np.hstack(trainD)
    voc = list(set(flat))

# counts by crp
def count_words(bounds):
    global counts
    words = Counter()
    words.update(np.hstack(apply_bounds(bounds)))
    counts.update(words)
    global cur_tot_words
    cur_tot_words = sum(counts.values())

def apply_bounds(bounds):
    boundD = []
    for b, s in zip(bounds, trainD):
        boundD.append([''.join(w) for w in np.array_split(s, b[:-1])])
    print(boundD)
    return boundD

'''
def dec_w_novel(nwrds):
    p_novel = a/(nwrds+a)
    p_not_novel = nwrds/(nwrds+a)
    return (p_novel >= p_not_novel)
'''

def prob_h1(word,final):
    nw1 = max(counts[word]-1, 0)

    p0_w = 1
    for c in list(word):
        p0_w *= p0[voc.index(c)]

    #return ((nw1+ a*p0_w)/(cur_tot_words -1 + a)) 

    nu = n if final else cur_tot_words-2
    return ((nw1+ a*p0_w)/(cur_tot_words -1 + a)) * ((nu + (rho/2))/(cur_tot_words-rho))

def prob_h2(word2,word3,final):
    nw2 = max(counts[word2]-1, 0)
    nw3 = max(counts[word3]-1, 0)

    p0_w2 = 1
    for c in list(word2):
        p0_w2 *= p0[voc.index(c)]

    p0_w3 = 1
    for c in list(word3):
        p0_w3 *= p0[voc.index(c)]

    #return ((nw2+ a*p0_w2)/(cur_tot_words-1 + a)) * ((nw3+ a*p0_w3)/(cur_tot_words-1 + a))
    
    iw = int(word2==word3)
    iu = int(np.invert(final)) #w2 and w3 can't both be utterance-final
    nu = n if final else cur_tot_words-2
    f1 = ((nw2+ a*p0_w2)/(cur_tot_words-1 + a)) * ((cur_tot_words-n+ (rho/2))/ (cur_tot_words+rho))
    f2 = ((nw3+iw+ a*p0_w3)/(cur_tot_words + a)) * ((nu+iu+ (rho/2))/ (cur_tot_words+rho))
    return f1*f2 
    

def precision(boundD):
    p = 0
    for b, d in zip(boundD, data):
        p += sum(np.in1d(b, d)) / float(len(d))
    return p/n

def recall(boundD):
    True

def f_measure(boundD):
    True


def gibbs(bounds):
    global save
    for e in range(epochs):
        print('epoch', e)
        for ut,bndrs in zip(trainD,bounds):
            b0 = 0
            for b in bndrs:
                final = b==len(ut)
                w1 = ''.join(ut[b0:b])
                p_h1 = prob_h1(w1,final)
                print(p_h1)
                for bj in range(b0,b):
                    w2 = ''.join(ut[b0:bj])
                    w3 = ''.join(ut[bj:b])
                    p_h2 = prob_h2(w2,w3,final)
                    print(p_h2)
                    #if p_h2 > p_h1:
                       #print('\t new boundary ')
    
    boundD = apply_bounds(bounds)
    if save:
        output = [' '.join(s) for s in boundD]
        with open('output_test','w') as out:
            out.write('\n'.join(output))

    # precision
    print('Precision:', precision(boundD))
    # recall 
    print('Recall:', recall(boundD))
    # f-measure
    print('F-Measure:', f_measure(boundD))

if __name__ == "__main__":
    args = parser.parse_args()
    n = args.trainN
    save = args.save
    a = args.alpha
    epochs = args.epochs
    rho = args.rho
    gibbs(load_data())
    
