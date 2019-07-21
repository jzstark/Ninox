#global: dictionary, docs
#type of model: z, nd, nw
#pull: subset of z, nd, nw
#push: diff of subset of z, nd, nw

import numpy as np
import random

"""
Configuration
"""

K = 10
alpha = 0.5
beta = 0.1
batch_size = 1000

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
    idx = random.randint(i * slicelen, i * slicelen + slicelen - 1)
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

    for d in range(D):
        l = len(docs[d])
        for i in range(l):
            topic = random.randint(0, K - 1)
            z[d].append(topic)

            tok_i = docs[d][i]
            nw[tok_i][topic] += 1
            nd[d][topic] += 1

    return z, nd, nw


def build_update():
    return {}, {}, {}


def update_model(model, u):
    (z0, nd0, nw0) = model
    (z1, nd1, nw1) = u
    for k, v in z1.items():
        z0[k] = np.add(z0[k], v)
    for k, v in nd1.items():
        nd0[k] = np.add(nd0[k], v)
    for k, v in nw1.items():
        nw0[k] = np.add(nw0[k], v)
    return (z0, nd0, nw0)


def compute_updates(model, i, n, step):
    z, nd, nw = model
    local_z, local_nd, local_nw = z.copy(), nd.copy(), nw.copy()

    docs_index = get_next_batch(i, n, step)
    for m in docs_index:
        for n in range(len(docs[m])):
            topic = local_z[m][n]
            w = docs[m][n]
            p = [0.0 for _ in range(K)]

            local_nw[w][topic] -= 1
            local_nd[m][topic] -= 1

            nwsum = [0] * K
            for i in nw.values():
                nwsum = np.add(nwsum, i)
            ndsum_m = len(docs[m])

            for k in range(K):
                p[k] = (local_nw[w][k] + beta) / (nwsum[k] + v_beta) * \
                    (local_nd[m][k] + alpha) / (ndsum_m + k_alpha)

            t = np.random.multinomial(1, np.divide(p, np.sum(p))).argmax()

            local_nw[w][t] += 1
            local_nd[m][t] += 1
            local_z[m][n] = t

    diff_z = {}; diff_nd = {}; diff_nw = {}
    for m in docs_index:
        diff_z[m]  = np.subtract(local_z[m], z[m])
        diff_nd[m] = np.subtract(local_nd[m], nd[m])
        for n in range(len(docs[m])):
            w = docs[m][n]
            diff_nw[w] = np.subtract(local_nw[w], nw[w])

    return diff_z, diff_nd, diff_nw


def compute_accuracy(model):
    local_z, local_nd, local_nw = model
    ll = 0.0
    nwsum = [0] * K
    for i in local_nw.values():
        nwsum = np.add(nwsum, i)

    for d, doc in enumerate(docs):
        ndsum_d = len(doc)
        for w in doc:
            l = np.divide(local_nw[w], nwsum) * \
                np.divide(local_nd[d], ndsum_d)
            ll = ll + np.log(np.sum(l))
    return None, ll


def test_run():
    N = 1
    opt = make_optimiser()
    model = build_model(opt, accuracy=True)
    for i in range(10):
        updates = compute_updates(model, i%N, N, i)
        model = update_model(model, updates)
        _, ll = compute_accuracy(model)
        print("Loglikelihood: %.2f" % ll)
