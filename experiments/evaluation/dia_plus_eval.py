#Erstmal wieder ROC AUC pro Klasse und Averaged
import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd
import pdb
from matplotlib import pyplot as plt


def get_data(base_path):
    # Load testset
    test_set = pd.read_csv(base_path + "DIA_PLUS_test_set.csv")

    # Load probs
    with open(base_path + "biobert_dia_plus_test_probabilities.txt") as probs_file:
        probs = probs_file.readlines()
    for i in range(len(probs)):
        probs[i] = probs[i].replace("\n", "")
        probs[i] = probs[i].split()

    # Add probs to test set
    test_set["probs"] = probs

    # Load all code names
    with open(base_path + "ALL_DIAGNOSES_PLUS_CODES.txt") as code_file:
        codes = code_file.read()
        codes = codes.split()

    test_set['y_true'] = test_set['labels']
    test_set['y_pred'] = test_set['probs']
    return test_set[['id', 'y_true', 'y_pred']], codes


def is_valid_digit_code(code, n_digits):
    """
        validate code has n_digits with two exceptions if the code starts with V or E and n_digits more digits
    """
    if code.startswith("V") or code.startswith("E"):
        code = code[1:]
    if len(code) == n_digits:
        try:
            int(code)
            return True
        except:
            return False

    return False


def get_mask(codes, dim):
    # get all valid positions for dim dimension i.e. 3 or 4 digit codes
    mask = [is_valid_digit_code(code, dim) for code in codes]
    return mask


def get_code_to_index(codes):
    # get map from code to column index of array
    return dict(tuple(zip(codes, np.arange(len(codes)))))


def get_code_predictions(df, bool_codes_mask):
    # get all dim digit positions from all predictions
    return np.vstack(df['y_pred'].apply(lambda x: np.array(x, dtype=float)[bool_codes_mask]).values)


def generate_labels(df, code_to_index, bool_codes_mask):
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
    return np.vstack(label_tmp)[:, bool_codes_mask]


def get_samples_per_code(y, rel_codes):
    samples_per_code = y.sum(axis=0)
    # create mapping from code to number of samples
    return dict([(code, samples) for code, samples in zip(rel_codes, samples_per_code)])


def get_score_per_code(y, yhat, codes, dim, min_samples):
    # get valid codes according to dim
    rel_codes = [code for code in codes if is_valid_digit_code(code, dim)]
    # create map valid code to relative column index (y contains already only valid code columns)
    rel_index_code_map = dict(tuple(zip(np.arange(len(rel_codes)), rel_codes)))
    # get only non empty columns
    cols = np.nonzero(y.sum(axis=0))[0]
    samples_per_code_map = get_samples_per_code(y, rel_codes)

    # calculate roc_auc_score per code
    res_dict = dict()
    for col in cols:
        code = rel_index_code_map[col]
        samples = samples_per_code_map[code]
        if code not in res_dict:
            res_dict[code] = dict()
        res_dict[code]['score'] = roc_auc_score(y[:, col], yhat[:, col], average='macro')
        res_dict[code]['#samples'] = samples

    # calculate total average score according to number of samples and top20 codes
    dict_list = list(res_dict.values())
    for min_sample in min_samples:
        vals = [x['score'] for x in dict_list if x['#samples'] > min_sample]
        print('minimum #samples', min_sample, 'average score', np.mean(vals))
        score_dict = dict([(code, dict_elem['score']) for code, dict_elem in res_dict.items() if dict_elem['#samples'] > min_sample])
        print(sorted(score_dict.items(), key=lambda item: item[1])[::-1][:20])


def code_to_group_mapping(group):
    group_dict = {
        1: {"start": 1, "end": 140},
        2: {"start": 140, "end": 240},
        3: {"start": 240, "end": 280},
        4: {"start": 280, "end": 290},
        5: {"start": 290, "end": 320},
        6: {"start": 320, "end": 390},
        7: {"start": 390, "end": 460},
        8: {"start": 460, "end": 520},
        9: {"start": 520, "end": 580},
        10: {"start": 580, "end": 630},
        11: {"start": 630, "end": 680},
        12: {"start": 680, "end": 710},
        13: {"start": 710, "end": 740},
        14: {"start": 740, "end": 760},
        15: {"start": 760, "end": 780},
        16: {"start": 780, "end": 800},
        17: {"start": 800, "end": 1000},
    }
    return group_dict[group]


def get_samples_per_group(y_group):
    # number of samples in group of codes
    return y_group.sum()


def get_score_per_group(y, yhat, codes, groups):
    # get valid group codes according
    rel_codes = [code for code in codes if is_valid_digit_code(code, n_digits=3)]
    # create map valid code to relative column index (y contains already only valid code columns)
    rel_code_index_map = dict(tuple(zip(rel_codes, np.arange(len(rel_codes)))))
    rel_index_code_map = dict(tuple(zip(np.arange(len(rel_codes)), rel_codes)))

    numeric_condition = lambda code: not code.startswith("V") and not code.startswith("E")
    non_numeric_condition = lambda code: code.startswith("V") or code.startswith("E")
    numeric_group_condition = lambda code: int(code) >= start and int(code) < end

    res_dict = dict()
    for group in groups:
        if group >= 1 and group <= 17:
            # get the right filter condition (start and end code) for each group
            condition_dict = code_to_group_mapping(group)
            start = condition_dict['start']
            end = condition_dict['end']
            # filter only numeric codes
            numeric_codes = [code for code in rel_codes if numeric_condition(code)]
            # filter codes for the specific group
            group_codes = dict([(code, rel_code_index_map[code]) for code in numeric_codes if numeric_group_condition(code)])
        elif group == 18:
            # filter non-numeric codes
            group_codes = dict([(code, rel_code_index_map[code]) for code in rel_codes if non_numeric_condition(code)])

        # calculate roc_auc_score per group
        group_cols = group_codes.values()
        non_zero_cols = np.nonzero(y.sum(axis=0))[0]
        # get only non-zero group columns
        non_zero_group_cols = list(set(group_cols).intersection(set(non_zero_cols)))
        if non_zero_group_cols:
            if group not in res_dict:
                res_dict[group] = dict()
            if group in [15]:
                res = list()
                for coltmp in non_zero_group_cols:
                    if rel_index_code_map[coltmp] == '776':
                        print(10*'+++', np.nonzero(y[:, coltmp]), 10*'+++')
                    auc = roc_auc_score(y[:, coltmp], yhat[:, coltmp], average='macro')
                    res.append(auc)
                    print('code',
                          rel_index_code_map[coltmp],
                          'samples',
                          y[:, coltmp].sum(),
                          'auc',
                          auc)
                print('group', group, 'average auc', np.mean(res))
                import pdb
                pdb.set_trace()
            samples_per_group = get_samples_per_group(y[:, non_zero_group_cols])
            res_dict[group]['score'] = roc_auc_score(y[:, non_zero_group_cols],
                                                     yhat[:, non_zero_group_cols],
                                                     average='macro')
            res_dict[group]['#samples'] = samples_per_group

        print('group', group, '#samples', res_dict[group]['#samples'], 'average score', res_dict[group]['score'])

    x = [res_dict[group]['#samples'] for group in groups]
    y = [res_dict[group]['score'] for group in groups]
    plt.scatter(x, y)
    plt.show()
    return res_dict


if __name__ == '__main__':

    dia_plus_path = "/home/micha/PycharmProjects/data/MIMIC_Predictions/"
    dim_var = 3
    label_df, codes = get_data(base_path=dia_plus_path)
    bool_codes_mask = get_mask(codes, dim=dim_var)
    y_pred = get_code_predictions(label_df, bool_codes_mask)
    # create mapping from code to column index
    code_to_index = get_code_to_index(codes)
    y_true = generate_labels(label_df, code_to_index, bool_codes_mask)
    get_score_per_code(y_true, y_pred, codes, dim=dim_var, min_samples=[0, 1, 5, 10, 25, 50, 100])
    if dim_var == 3:
        group_score = get_score_per_group(y_true, y_pred, codes, groups=range(1, 19))



    """
        Für die Analyse fände ich vor allem interessant:
        Wie gut ist das Modell auf 4-digit Vorhersage? 0.7879 545283542114
        Wie gut ist das Modell auf Übergruppen-Vorhersage? 
        Dafür musst du die Codes in ihre Übergruppen zurück führen. 
        Dafür kannst du Teile aus dem dia_groups.py Skript im experiments/dia Ordner wieder verwenden, da sind die Gruppen schon eingeteilt.
        Auf welchen 20 Codes funktioniert das Modell am besten?
    """

