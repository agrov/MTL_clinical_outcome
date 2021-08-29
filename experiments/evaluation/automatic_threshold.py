import os

from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import pdb
import pickle
from tqdm import tqdm


def get_data(test_set, pred_set, all_codes_file):
    # Load testset
    test_set = pd.read_csv(test_set)

    # Load probs
    with open(pred_set) as probs_file:
        probs = probs_file.readlines()
    for i in range(len(probs)):
        probs[i] = probs[i].replace("\n", "").split()

    # Add probs to test set
    test_set["probs"] = probs

    # Load all code names
    with open(all_codes_file) as code_file:
        codes = code_file.read()
        codes = codes.split()

    test_set['y_true'] = test_set['labels']
    test_set['y_pred'] = test_set['probs']

    return test_set, codes


def get_code_predictions(df):
    # get all dim digit positions from all predictions
    return np.vstack(df['y_pred'].apply(lambda x: np.array(x, dtype=float)).values)


def generate_labels(df, code_to_index):
    label_tmp = []
    for index, row in df.iterrows():
        y_true = row.y_true.split(',')
        # get column indices of filtered codes
        # get index for all codes in every row of y_true except ''
        indices = np.array([code_to_index[dia] for dia in y_true if dia != ''])
        # initialize label vector
        label = np.zeros((len(code_to_index),), dtype=float)
        if len(indices):
            # set all indices of occuring codes to 1 for every row in y_true
            label[indices] = 1.0
        label_tmp.append(label)
    # filter only dim digit valid indices/columns
    return np.vstack(label_tmp)


def geometric_mean(y, yhat):
    fpr, tpr, thresholds = roc_curve(y, yhat)

    # calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1 - fpr))

    # locate the index of the largest g-mean
    ix = np.argmax(gmeans)

    # print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
    return thresholds[ix]


def precision_recall(y, yhat):
    precision, recall, thresholds = precision_recall_curve(y, yhat)
    # convert to f score
    fscore = (2 * precision * recall) / (precision + recall)
    # locate the index of the largest f score
    ix = np.argmax(fscore)
    # print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
    return thresholds[ix]


def optimal_threshold(y, yhat):
    thresholds = np.arange(0, 1, 0.01)

    # apply threshold to positive probabilities to create labels
    def to_labels(pos_probs, threshold):
        return (pos_probs >= threshold).astype('int')

    # evaluate each threshold
    scores = [f1_score(y, to_labels(yhat, t)) for t in thresholds]
    # get best threshold
    ix = np.argmax(scores)
    # print('Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))
    return thresholds[ix]


def create_thresholds(y, yhat, basepath, strategy):
    print('START threshold calculations')
    nonzero_cols = np.nonzero(y.sum(axis=0))[0]
    thresholds = dict()
    print('threshold determined by', strategy)
    for code in tqdm(codes):
        col = code_to_index[code]
        if col in nonzero_cols:
            if strategy == 'geometric_mean':
                thresholds[code] = geometric_mean(y[:, col], yhat[:, col])
            elif strategy == 'precision_recall':
                thresholds[code] = precision_recall(y[:, col], yhat[:, col])
            elif strategy == 'optimal_threshold':
                thresholds[code] = optimal_threshold(y[:, col], yhat[:, col])
            else:
                print('choose a VALID strategy')
                raise
        else:
            thresholds[code] = np.nan

    with open(os.path.join(basepath, f'{strategy}_thresholds.pcl'), 'wb') as f:
        pickle.dump(thresholds, f, pickle.HIGHEST_PROTOCOL)


def get_code_maps(codes):
    # get map from code to column index of array
    return dict(tuple(zip(codes, np.arange(len(codes))))), dict(tuple(zip(np.arange(len(codes)), codes)))


def save_all_thresholds(y, yhat, strategies, base_path):
    for strategy in strategies:
        create_thresholds(y, yhat, base_path, strategy=strategy)


def read_thresholds(strategy, basepath):
    with (open(os.path.join(basepath, f'{strategy}_thresholds.pcl'), "rb")) as f:
        thresholds = pickle.load(f)
    return thresholds


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--all_codes_file")
    parser.add_argument("--pred_file")
    parser.add_argument("--test_file")
    parser.add_argument("--save_dir")

    return parser.parse_args()


if __name__ == '__main__':
    strategies = ['geometric_mean', 'precision_recall', 'optimal_threshold']

    args = get_args()
    label_df, codes = get_data(args.test_file, args.pred_file, args.all_codes_file)
    y_pred = get_code_predictions(label_df)
    code_to_index, index_to_code = get_code_maps(codes)
    y_true = generate_labels(label_df, code_to_index)
    save_all_thresholds(y_true, y_pred, strategies, args.save_dir)
