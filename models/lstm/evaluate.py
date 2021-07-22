import numpy.core.records
import numpy as np

def evaluete_fn(predict_func, sequences_test: np.array, labels_test: np.array) -> dict:
    return {
        'acc': 0,
        'precision': 0,
        'recal': 0,
        'f1': 1,
        'auc': 1
    }