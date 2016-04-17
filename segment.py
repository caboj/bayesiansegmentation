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
burnin = 0

def load_data():
    global data
    global trainD
    global n

    data = []
    with open('data/br-phono-train.txt', 'rU') as f:
        if not n:
            for l in f:
                data.append((l.strip()).split(' '))
        else:
            data = np.array([np.array( (f.readline().strip()).split(' ') ) for i in range(n)])
    trainD = np.array([np.array(list(''.join(s))) for s in data])
    n = len(trainD)
    #print('len data: ', n)

    n_uf = n
    # initialize boundries between words
    bounds = [[len(trainD[i])] for i in range(len(trainD))]
    #bounds = [[i for i in range(1,len(ut))] for ut in trainD]
    for i in range(n):
        l = len(trainD[i])
        if l > 2:
            nrBs = int(l/4)+1
            bounds[i] = np.append( np.sort(np.random.choice(np.arange(1,l),replace=False,size=(nrBs))),bounds[i]).tolist()
    
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
    #totalCs = len(np.hstack(trainC))

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
    #print(boundD)
    return boundD

def prob_h1(word,final,new):
    tot_words = cur_tot_words - 1 if new else cur_tot_words - 2
    nw1 = max(counts[word]-1, 0)

    p0_w = 1
    for c in list(word):
        p0_w *= p0[voc.index(c)]

    #return ((nw1+ a*p0_w)/(cur_tot_words -1 + a)) 

    nu = n if final else tot_words-n
    return ((nw1+ a*p0_w)/(tot_words + a)) * ((nu + (rho/2))/(tot_words+rho))

def prob_h2(word2,word3,final,new):
    tot_words = cur_tot_words - 1 if new else cur_tot_words - 2
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
    nu = n if final else tot_words - n
    min1 = 1 if final else 0
    f1 = ((nw2+a*p0_w2)/(tot_words + a)) * ((tot_words-n-min1+ (rho/2))/ (tot_words+rho))
    f2 = ((nw3+iw+a*p0_w3)/(tot_words + a)) * ((nu+iu+(rho/2))/ (tot_words+rho))
    return f1*f2
    

def extract_bounds():
    bounds = []
    for ut in data:
        lengths = [len(w) for w in ut]
        bounds.append([sum(lengths[0:i+1]) for i in range(len(lengths))])
    return bounds

def precision(foundB, dataB):
    p = np.array([])
    for b, utb in zip(foundB, dataB):
        denom = max(float(len(b)), 1.0)
        pb = sum(np.in1d(b, utb)) / denom
        p = np.append(p, pb)
    return p

def recall(foundB, dataB):
    r = np.array([])
    for b, utb in zip(foundB, dataB):
        #print(b)
        #print(utb)
        #print(np.in1d(b, utb))
        denom = max(float(len(utb)), 1.0)
        rb = sum(np.in1d(b, utb)) / denom
        #print(rb)
        #print('')
        r = np.append(r, rb)
    return r

def f_measure(p, r):
    sum_pr = p+r
    sum_pr[sum_pr==0]=1
    f = 2*((p*r)/sum_pr)
    return f

def test_h1_gr_h2(b0,cur_b,end_b,ut,new):
    final = end_b==len(ut)
    w1 = ''.join(ut[b0:end_b])
    w2 = ''.join(ut[b0:cur_b])
    w3 = ''.join(ut[cur_b:end_b])

    p_h1 = prob_h1(w1,final,new)
    p_h2 = prob_h2(w2,w3,final,new)

    #print('w1: ',w1,'w2: ', w2, 'w3: ',w3)
    r = np.random.uniform(0,1)
    if p_h1 > p_h2 and r < p_h2/p_h1:
        return True
    else:
        return False
    if p_h1 < p_h2 and r < p_h1/p_h2:
        return False
    else:
        return True


def update_counts_remove(ut, b_idx, bndrs):
    b0 = 0
    if b_idx>0:
        b0 = bndrs[b_idx-1]
    bn = len(ut)-1
    if b_idx<len(bndrs)-1:
        bn = bndrs[b_idx+1]
    bi = bndrs[b_idx]
    
    w = np.array_split(ut, [b0, bi, bn])
    w2 = ''.join(w[1])
    w3 = ''.join(w[2])
    w1 = w2+w3
    
    global counts
    #decrease
    counts[w2] = counts[w2]-1
    counts[w3] = counts[w3]-1
    #increase
    counts[w1] = counts[w1]+1

def update_counts_add(ut, b_idx, bi, bndrs):
    b0 = 0
    if b_idx>0:
        b0 = bndrs[b_idx-1]
    bn = len(ut)-1
    if b_idx<len(bndrs)-1:
        bn = bndrs[b_idx]

    w = np.array_split(ut, [b0, bi, bn])
    w2 = ''.join(w[1])
    w3 = ''.join(w[2])
    w1 = w2+w3

    global counts
    #increase
    counts[w2] = counts[w2]+1
    counts[w3] = counts[w3]+1
    #decrease
    counts[w1] = counts[w1]-1

def word_eval_bounds(bounds):
    wBounds = []
    for b in bounds:
        b = np.append([0],b)
        wbs = ['-'.join(b.astype('str')[i:i+2]) for i in range(len(b)-1)]
        wBounds.append(wbs)
    return wBounds

def word_eval_lexicon(utterances):
    lex = []
    for ut in utterances:
        lex.append(list(set(ut)))
    return lex

def word_eval_ambiguous(bounds):
    aBounds = []
    for b in bounds:
        aBounds.append(b[:-1])
    return aBounds

def evaluate(bounds, dataB, boundD):
    boundsW = word_eval_bounds(bounds) 
    dataBW = word_eval_bounds(dataB)
    # precision
    p = precision(boundsW, dataBW)
    print('P:', sum(p)/n)
    # recall 
    r = recall(boundsW, dataBW)
    print('R:', sum(r)/n)
    # f-measure
    f = f_measure(p,r)
    print('F:', sum(f)/n)
   
    dataL = word_eval_lexicon(data)
    boundDL = word_eval_lexicon(boundD)
    # precision
    pl = precision(boundDL, dataL)
    print('\nLP:', sum(pl)/n)
    # recall 
    rl = recall(boundDL, dataL)
    print('LR:', sum(rl)/n)
    # f-measure
    fl = f_measure(pl,rl)
    print('LF:', sum(fl)/n)

    boundsA = word_eval_ambiguous(bounds)
    dataBA = word_eval_ambiguous(dataB)
    # precision
    pb = precision(boundsA, dataBA)
    print('\nBP:', sum(pb)/n)
    # recall 
    rb = recall(boundsA, dataBA)
    print('BR:', sum(rb)/n)
    # f-measure
    fb = f_measure(pb,rb)
    print('BF:', sum(fb)/n)

def gibbs(bounds):
    global save
    
    for e in range(epochs):
        print('epoch', e, end='\r')
        utI = 0
        b_changes = 0
        for ni in range(n):
            #bndrs = bndrs.tolist()
            existing_b = False
            ut = trainD[ni]
            #print(ut)
            bndrs = bounds[ni]
            #print(bndrs)
            b0 = 0
            end_b = bndrs[0]
            end_b_idx = 0
            for cur_b in range(1,len(ut)-1):
                if cur_b == end_b:
                    end_b_idx += 1
                    end_b = bndrs[end_b_idx]
                    h1 = test_h1_gr_h2(b0,cur_b,end_b,ut,False)
                    if h1:
                        b_changes += 1
                        #print('remove b, b_idx: ',cur_b,end_b_idx)
                        update_counts_remove(ut, end_b_idx-1, bndrs)
                        if e > burnin:
                            end_b_idx -= 1
                            bndrs = bndrs[:end_b_idx]+bndrs[end_b_idx+1:]
                    else:
                        b0 = bndrs[end_b_idx-1]

                else:
                    h1 = test_h1_gr_h2(b0,cur_b,end_b,ut,True)
                    if not h1:
                        b_changes += 1
                        #print('insert b, b_idx: ', cur_b,end_b_idx)
                        update_counts_add(ut, end_b_idx, cur_b, bndrs)
                        if e > burnin:
                            bndrs.insert(end_b_idx,cur_b)
                            end_b_idx += 1
                        b0 = cur_b
            bounds[ni] = bndrs
        print('\t\tboundaries canged: ', b_changes,end='\r')
    # bounds according to data
    dataB = extract_bounds()
    # segmented utterances according to found bounds
    boundD = apply_bounds(bounds)
    if save:
        output = [' '.join(s) for s in boundD]
        with open('output_test','w') as out:
            out.write('\n'.join(output))

    # calculate evaluation metrics
    evaluate(bounds, dataB, boundD)

if __name__ == "__main__":
    args = parser.parse_args()
    n = args.trainN
    save = args.save
    a = args.alpha
    epochs = args.epochs
    rho = args.rho
    gibbs(load_data())
    
