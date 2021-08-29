
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
from argparse import Namespace, ArgumentParser
import numpy as np
from datasets import load_dataset, load_metric
from transformers.integrations import MLflowCallback
from sklearn.metrics import roc_auc_score
import transformers
import torch
import ray
from ray import tune
from hyperopt import hp
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from transformers import (
    AdapterConfig,
    AdapterType,
    AutoConfig,
    AutoModelWithHeads,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    MultiLingAdapterArguments,
    PretrainedConfig,
    # Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    TextClassificationPipeline
)
from transformers.trainer_utils import is_main_process
from numpy import exp
import pandas as pd

task_to_keys = {
    "dia": "text",
    "pro": "text",
    "los": "text",
    "mp": "text"
}

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda")

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    label_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the file including all the labels [valid only for multilabel tasks]: " + ", ".join(
                task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    test_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            extension = self.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )



def eval_model(config_file,model_dir, test_file):
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, MultiLingAdapterArguments))
    model_args, data_args, training_args, adapter_args = parser.parse_json_file(json_file=config_file)

    if data_args.task_name == 'dia' or data_args.task_name == 'pro':
        task_type = 'multilabel'
    elif data_args.task_name == 'mp':
        task_type = 'binary'
    else:
        task_type = 'multiclass'

    if data_args.task_name is not None:
        if task_type == 'multilabel':
            with open(data_args.label_file) as code_file:
                label_list = code_file.read().split(" ")
                label_list.sort()
                num_labels = len(label_list)
        elif data_args.task_name=="mp":
            num_labels=2
        else:
            num_labels=4

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )
    model = AutoModelWithHeads.from_pretrained(model_args.model_name_or_path,
                                               from_tf=bool(".ckpt" in model_args.model_name_or_path),
                                               config=config,
                                               cache_dir=model_args.cache_dir,
                                               )
    adapter_name = model.load_adapter(model_dir, config="pfeiffer")
    model.set_active_adapters(adapter_name)
    tc = TextClassificationPipeline(model=model, tokenizer=tokenizer)
    test_df = pd.read_csv(test_file, usecols=["text"])
    for index,row in test_df.iterrows():
        result = tc(row['text'])
        print("💡", result["label"])

if __name__ == "__main__":
    eval_model(config_file="/home/anjali/MTL/experiments/Adapters/config/local_args.json",model_dir="/home/anjali/MTL/experiments/models/adapter_mp/mp",test_file="filtered_ids_inference.csv")