import os
import csv
import numpy as np

from tqdm import tqdm

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

seed = 3535999445

def _rocstories(path):
    with open(path, encoding='utf_8') as f:
        f = csv.reader(f)
        st = []
        ct1 = []
        ct2 = []
        y = []
        for i, line in enumerate(tqdm(list(f), ncols=80, leave=False)):
            if i > 0:
                s = ' '.join(line[1:5])
                c1 = line[5]
                c2 = line[6]
                st.append(s)
                ct1.append(c1)
                ct2.append(c2)
                y.append(int(line[-1])-1)
        return st, ct1, ct2, y

def rocstories(data_dir, n_train=1497, n_valid=374):
    storys, comps1, comps2, ys = _rocstories(os.path.join(data_dir, 'cloze_test_val__spring2016 - cloze_test_ALL_val.csv'))
    teX1, teX2, teX3, _ = _rocstories(os.path.join(data_dir, 'cloze_test_test__spring2016 - cloze_test_ALL_test.csv'))
    tr_storys, va_storys, tr_comps1, va_comps1, tr_comps2, va_comps2, tr_ys, va_ys = train_test_split(storys, comps1, comps2, ys, test_size=n_valid, random_state=seed)
    trX1, trX2, trX3 = [], [], []
    trY = []
    for s, c1, c2, y in zip(tr_storys, tr_comps1, tr_comps2, tr_ys):
        trX1.append(s)
        trX2.append(c1)
        trX3.append(c2)
        trY.append(y)

    vaX1, vaX2, vaX3 = [], [], []
    vaY = []
    for s, c1, c2, y in zip(va_storys, va_comps1, va_comps2, va_ys):
        vaX1.append(s)
        vaX2.append(c1)
        vaX3.append(c2)
        vaY.append(y)
    trY = np.asarray(trY, dtype=np.int32)
    vaY = np.asarray(vaY, dtype=np.int32)
    return (trX1, trX2, trX3, trY), (vaX1, vaX2, vaX3, vaY), (teX1, teX2, teX3)

# MSR Paraphrase Corpus
def _msrpc(path):
    with open(path, encoding='utf_8') as f:
        f = csv.reader(f, delimiter='\t')
        s1 = [] # string 1
        s2 = [] # string 2
        y = []  # label/quality (0 or 1)
        for i, line in enumerate(tqdm(list(f), ncols=80, leave=False)):
            if i > 0: # skip header line
                s1.append(line[3])
                s2.append(line[4])
                y.append(line[0])
        return s1, s2, y

def msrpc(data_dir, n_train=1497, n_valid=374):
    sent1, sent2, ys = _msrpc(os.path.join(data_dir, 'msr_paraphrase_train.txt')) # complete training data
    teS1, teS2, teY = _msrpc(os.path.join(data_dir, 'msr_paraphrase_test.txt')) # test
    tr_sent1, va_sent1, tr_sent2, va_sent2, tr_ys, va_ys = train_test_split(sent1, sent2, ys, test_size=n_valid, random_state=seed) # break the training data into training and validation splits
    trS1, trS2 = [], []
    trY = []
    for s1, s2, y in zip(tr_sent1, tr_sent2, tr_ys):
        trS1.append(s1)
        trS2.append(s2)
        trY.append(y)

    vaS1, vaS2 = [], []
    vaY = []
    for s1, s2, y in zip(va_sent1, va_sent2, va_ys):
        vaS1.append(s1)
        vaS2.append(s2)
        vaY.append(y)
    trY = np.asarray(trY, dtype=np.int32)
    vaY = np.asarray(vaY, dtype=np.int32)
    return (trS1, trS2, trY), (vaS1, vaS2, vaY), (teS1, teS2, teY)