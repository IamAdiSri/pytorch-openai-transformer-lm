import os
import json
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score

from datasets import _msrpc

def msrpc(data_dir, pred_path, log_path):
    preds = pd.read_csv(pred_path, delimiter='\t')['prediction'].values.tolist()
    _, _, labels = _msrpc(os.path.join(data_dir, 'msr_paraphrase_test.txt'))
    test_accuracy = accuracy_score(labels, preds)*100.
    logs = [json.loads(line) for line in open(log_path)][1:]
    best_validation_index = np.argmax([log['va_acc'] for log in logs])
    valid_accuracy = logs[best_validation_index]['va_acc']
    print('MSR Paraphrase Corpus Valid Accuracy: %.2f'%(valid_accuracy))
    print('ROC Paraphrase Corpus Test Accuracy:  %.2f'%(test_accuracy))
