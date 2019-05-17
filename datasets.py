import os
import csv
import numpy as np

from tqdm import tqdm

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

seed = 3535999445

# MSR Paraphrase Corpus
def _msrpc(path):
    with open(path, encoding='utf_8') as f:
        f = f.readlines()
        for i, line in enumerate(f):
            f[i] = f[i].strip().split('\t')
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
    teS1, teS2, _ = _msrpc(os.path.join(data_dir, 'msr_paraphrase_test.txt')) # test
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
    return (trS1, trS2, trY), (vaS1, vaS2, vaY), (teS1, teS2)