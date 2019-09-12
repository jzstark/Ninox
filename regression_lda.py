#global: dictionary, docs
#type of model: z, nd, nw
#pull: subset of z, nd, nw
#push: diff of subset of z, nd, nw

import numpy as np
import random
import time
from joblib import Parallel, delayed

"""
Configuration
"""

K = 10


class LDADictionary:
    def __init__(self):
        self.word2id = dict()
        self.id2word = dict()
        self.stop_list = set('for a of the and to in'.split())

    def num_words(self):
        return len(self.word2id)

    def get_word(self, word_id):
        return self.id2word[word_id]

    def get_id(self, word):
        return self.word2id[word]

    def contains_word(self, word):
        return True if word in self.word2id else False

    def contains_id(self, word_id):
        return True if word_id in self.id2word else False

    def add_word(self, word):
        if not self.contains_word(word) and word not in self.stop_list:
            word_id = len(self.word2id)
            self.word2id[word] = word_id
            self.id2word[word_id] = word
            return word_id
        else:
            return self.get_id(word)

"""
Preprocess Data
"""
#====================
# Dataset 1

dictionary = LDADictionary()
raw_docs = []
with open('data/lda_data.txt', 'r') as f:
    for line in f:
        raw_docs.append(line)
D = int(raw_docs[0])
raw_docs = raw_docs[1:]
docs =[None] * D

def set_doc(raw_str, idx):
    words = raw_str.lower().split()
    tokens = []
    for w in words:
        if dictionary.contains_word(w):
            wid = dictionary.get_id(w)
        else:
            wid = dictionary.add_word(w)
        tokens.append(wid)
    docs[idx] = tokens


for idx, raw_doc in enumerate(raw_docs):
    set_doc(raw_doc, idx)

W = dictionary.num_words()

# W = 8211, D = 10768

#====================


#====================
# Dataset 2
# Src: https://towardsdatascience.com/topic-modeling-for-the-new-york-times-news-dataset-1f643e15caac

"""
docs = []
with open('data/lda_nyt_small.txt', 'r') as f:
    for l in f.readlines():
        line = [int(x.split(':')[0]) - 1 for x in l.split(',')]
        docs.append(line)
D = 8447
W = 3012
"""

#====================

#====================
# Dataset 3: Blei
# Src: http://www.cs.columbia.edu/~blei/lda-c/

"""
docs = []
with open('data/lda_ap.txt', 'r') as f:
    for l in f.readlines():
        line = [int(x.split(':')[0]) for x in l.split(' ')[1:]]
        docs.append(line)
D = 2246
W = 10473
"""

#====================

batch_size = int(D / 32)
alpha = 50. / K
beta = 200. / W
# Reference: https://blog.csdn.net/pipisorry/article/details/42129099

v_beta = float(W * beta)
k_alpha = float(K * alpha)

"""
Utilities
"""
train_len = len(docs)

def get_next_batch(i, n, clock):
    size = batch_size
    slicelen = int((train_len - size - 1) / n)
    #idx = i * slicelen + (clock * data_select_step) % slicelen
    #idx = random.randint(i * slicelen, i * slicelen + slicelen - 1)
    idx = i * slicelen
    return list(range(idx, idx+size+1))


"""
Interfaces
"""

def make_optimiser():
    return None


def build_model(opt, accuracy=True):
    nd = dict()
    nw = dict()
    z  = dict()

    t_init = [0 for x in range(K)]
    for i in range(D): nd[i] = t_init.copy()
    for i in range(W): nw[i] = t_init.copy()
    for i in range(D): z[i]  = []
    nwsum = [0 for x in range(K)]

    for d in range(D):
        l = len(docs[d])
        for i in range(l):
            topic = random.randint(0, K - 1)
            z[d].append(topic)

            tok_i = docs[d][i]
            nw[tok_i][topic] += 1
            nd[d][topic] += 1
            nwsum[topic] += 1
    return z, nd, nw, nwsum


def build_update():
    return {}, {}, {}, [0] * K


def update_model(model, u):
    (z0, nd0, nw0, nws0) = model
    (z1, nd1, nw1, nws1) = u
    for k, v in z1.items():
        z0[k] = np.add(z0[k], v)
    for k, v in nd1.items():
        nd0[k] = np.add(nd0[k], v)
    for k, v in nw1.items():
        nw0[k] = np.add(nw0[k], v)
    nws0 = np.add(nws0, nws1)
    return (z0, nd0, nw0, nws0)


def compute_updates(model, i, n, step):
    z, nd, nw, nwsum = model
    local_z, local_nd, local_nw, local_nwsum = z.copy(), nd.copy(), nw.copy(), nwsum.copy()

    docs_index = get_next_batch(i, n, step)
    for m in docs_index:
        ndsum_m = len(docs[m])
        for n in range(len(docs[m])):
            topic = local_z[m][n]
            w = docs[m][n]

            local_nw[w][topic] -= 1
            local_nd[m][topic] -= 1
            local_nwsum[topic] -= 1

            p = [0.0 for _ in range(K)]
            for k in range(K):
                #p[k] = (local_nw[w][k] + beta) / (nwsum[k] + v_beta) * \
                p[k] = (local_nw[w][k] + beta) / (local_nwsum[k] + v_beta) * \
                    (local_nd[m][k] + alpha) / (ndsum_m + k_alpha)
            t = np.random.multinomial(1, np.divide(p, np.sum(p))).argmax()

            local_nw[w][t] += 1
            local_nd[m][t] += 1
            local_nwsum[t] += 1
            local_z[m][n] = t

    diff_z = {}; diff_nd = {}; diff_nw = {}
    diff_nwsum = np.subtract(local_nwsum, nwsum)

    for m in docs_index:
        diff_z[m]  = np.subtract(local_z[m], z[m])
        diff_nd[m] = np.subtract(local_nd[m], nd[m])
        for n in range(len(docs[m])):
            w = docs[m][n]
            diff_nw[w] = np.subtract(local_nw[w], nw[w])

    return diff_z, diff_nd, diff_nw, diff_nwsum


def compute_accuracy(model):
    local_z, local_nd, local_nw, local_nwsum = model
    ll = 0.0
    for d, doc in enumerate(docs):
        ndsum_d = len(doc)
        div_nd = np.divide(local_nd[d], ndsum_d)
        for w in doc:
            l = np.divide(local_nw[w], local_nwsum) * div_nd
            ll = ll + np.log(np.sum(l))
    return None, ll


def test_run():
    N = 1
    opt = make_optimiser()
    model = build_model(opt, accuracy=True)
    for i in range(10):
        start = time.time()
        updates = compute_updates(model, i%N, N, i)
        end1 = time.time()
        model = update_model(model, updates)
        end2 = time.time()
        _, ll = compute_accuracy(model)
        end3 = time.time()
        print(end1 - start, end2 - start, end3 - start)
        # (2s, 2s, 14s) --> compute_accuracy is slow, but nothing we could do.
        print("Loglikelihood: %.2f" % ll)
