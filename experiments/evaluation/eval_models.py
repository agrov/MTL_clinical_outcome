import argparse
import logging
import pandas as pd

from farm.infer import Inferencer
import numpy as np

logger = logging.getLogger(__name__)


def eval_lm(args):
    model_dir = args.model_dir
    test_file = args.test_file

    texts = pd.read_csv(test_file, usecols=["text"])
    dicts = []
    for i, row in texts.iterrows():
        dicts.append({"text": row["text"]})

    inferencer = Inferencer.load(str(model_dir), task_type="embeddings", num_processes=0)
    results = inferencer.extract_vectors(dicts)

    vec_array = [result["vec"] for result in results]
    vecs = np.stack(vec_array, axis=0)

    np.savetxt('vectors.tsv', vecs, delimiter='\t')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True, help='model dir')
    parser.add_argument('--test_file', required=True, help='test file')

    eval_lm(parser.parse_args())
