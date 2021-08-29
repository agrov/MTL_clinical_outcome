import argparse
import logging
import pandas as pd
import os
from farm.infer import Inferencer
import numpy as np

logger = logging.getLogger(__name__)

"""Infer FARM model on Diagnosis task"""
def eval_lm(model_dir,test_file):
    # model_dir = args.model_dir
    # test_file = args.test_file
    texts = pd.read_csv(test_file, usecols=["text"])
    dicts = []
    for i, row in texts.iterrows():
        dicts.append({"text": row["text"]})

    inferencer = Inferencer.load(str(model_dir), task_type="text_classification",num_processes=0,return_class_probs=True)
    results = inferencer.inference_from_dicts(dicts, return_json=True)
    results
    print(results[0][0]['predictions'])
    with open(os.path.join(str(model_dir), "results_inference.txt"), "w") as output:
        output.write(str(results))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True, help='model dir')
    parser.add_argument('--test_file', required=True, help='test file')
    #model_dir="/data_dir/MTL/experiments/models/dia_pro/outcome_multitask_dia_pro_final/exp_336472"
    #test_file="filtered_ids_inference.csv"
    eval_lm(model_dir,test_file)
