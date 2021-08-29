import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np
import logging
import argparse

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO)
logger = logging.getLogger(__name__)


def roc_auc_from_predictions(preds_file, test_file, all_codes_file, medcat_file, test_file_label="labels",
                             medcat_policy="all"):
    def is_3_digit_code(code):
        if code.startswith("V") or code.startswith("E"):
            code = code[1:]
        if len(code) == 3:
            try:
                int(code)
                return True
            except:
                return False
        return False

    def binary_classification_metrics_3_digits_only(probs, labels, all_codes):
        mask = list(map(is_3_digit_code, all_codes))
        print(f"Evaluate on {mask.count(True)} 3-digit-codes.")

        return [prob[mask] for prob in probs], labels[:, mask]

    # Read ALL CODES
    with open(all_codes_file, "r") as cf:
        all_codes_text = cf.read().strip()

    all_codes = all_codes_text.split(" ")

    # Read TEST FILE
    test_df = pd.read_csv(test_file)

    if medcat_policy == "all":
        label_text_array = test_df[test_file_label].to_numpy()
    else:
        label_text_array = extract_medcat_labels_from_test(test_df, medcat_file, medcat_policy)
    test_rows = []
    for row in label_text_array:
        row_vec = np.zeros(len(all_codes))

        # split labels in row
        row_labels = row.split(",")
        for label in row_labels:
            if label != "":
                row_vec[all_codes.index(label)] = 1
        test_rows.append(row_vec)
    labels = np.array(test_rows, dtype=int)

    # Read PREDS FILE
    with open(preds_file, "r") as pf:
        preds_lines = pf.readlines()

    probs = []
    for line in preds_lines:
        prob_row = line.replace("\n", "").split(" ")
        prob_row = [float(x) for x in prob_row]
        probs.append(np.array(prob_row))

    logger.info(f"Dim Predictions: {len(probs[0])}")
    logger.info(f"Length Predictions: {len(probs)}")

    # Reduce to 3-DIGITS
    probs, labels = binary_classification_metrics_3_digits_only(probs, labels, all_codes)

    # Remove empty cols
    dim_size = len(labels[0])
    mask = np.ones((dim_size), dtype=bool)
    for c in range(dim_size):
        if max(labels[:, c]) == 0:
            mask[c] = False
    labels = labels[:, mask]
    y_score = np.array(probs)[:, mask]

    filtered_cols = np.count_nonzero(mask == False)
    logger.info(f"{filtered_cols} columns not considered for ROC AUC calculation!")

    # Calculate ROC AUC
    roc = roc_auc_score(labels, y_score, average="macro")

    logger.info(f"ROC AUC for Medcat '{medcat_policy}': {roc}")


def extract_medcat_labels_from_test(test_df, medcat_file, medcat_policy):
    medcat_df = pd.read_csv(medcat_file, delimiter=";", usecols=["HADM_ID", "ICD9_CODE", "MedCatPredicted"])
    test_df["extracted_labels"] = ""

    if medcat_policy == "extracted_only":
        medcat_filter = "Yes"
    else:
        medcat_filter = "No"

    for i, row in medcat_df[medcat_df.MedCatPredicted == medcat_filter].iterrows():
        if test_df[test_df.id == row["HADM_ID"]].extracted_labels.item() == "":
            test_df.loc[test_df.id == row["HADM_ID"], "extracted_labels"] = row["ICD9_CODE"]
        else:
            test_df.loc[test_df.id == row["HADM_ID"], "extracted_labels"] = test_df[test_df.id == row[
                "HADM_ID"]].extracted_labels.item() + "," + row["ICD9_CODE"]

    return test_df.extracted_labels.to_numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds")
    parser.add_argument("--codes",
                        default="/persist/bvanaken/outcome-prediction/tasks/dia/data/ALL_DIAGNOSES_PLUS_CODES.txt")
    parser.add_argument("--test_file",
                        default="/persist/bvanaken/outcome-prediction/tasks/dia/data/DIA_PLUS_adm_test.csv")
    parser.add_argument("--test_file_label", default="labels")
    parser.add_argument("--medcat_file",
                        default="/persist/bvanaken/outcome-prediction/experiments/evaluation/DIA_MIMIC_MEDCAT.csv")
    args = parser.parse_args()
    roc_auc_from_predictions(args.preds, args.test_file, args.codes, args.medcat_file,
                             test_file_label=args.test_file_label,
                             medcat_policy="all")
    roc_auc_from_predictions(args.preds, args.test_file, args.codes, args.medcat_file,
                             test_file_label=args.test_file_label,
                             medcat_policy="extracted_only")
    roc_auc_from_predictions(args.preds, args.test_file, args.codes, args.medcat_file,
                             test_file_label=args.test_file_label,
                             medcat_policy="non_extracted_only")
