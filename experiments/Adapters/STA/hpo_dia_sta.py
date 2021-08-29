
""" HPO for Single Task Adapters- Diagnosis"""

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
#from AdapterEarlyStopping import EarlyStoppingCallback,ReportCallback
from trainer_callback import DefaultFlowCallback,EarlyStoppingCallback
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
    #Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import is_main_process
from numpy import exp
import pandas as pd
from Adapters.Extended_Trainer import ExtendedTrainer

task_to_keys = {
    "dia": "text",
    "pro":"text",
    "los":"text",
    "mp":"text"
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
        metadata={"help": "The name of the file including all the labels [valid only for multilabel tasks]: " + ", ".join(task_to_keys.keys())},
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


def doc_classification(args,config_file,run_name,do_hpo=False,multilabel=False):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, MultiLingAdapterArguments))
    model_args, data_args, training_args,adapter_args = parser.parse_json_file(json_file=config_file)
    batch_size = 20
    training_args.evaluation_strategy="steps"
    training_args.learning_rate=1e-5
    training_args.per_device_train_batch_size=batch_size
    training_args.per_device_eval_batch_size=batch_size
    training_args.num_train_epochs=200
    training_args.eval_steps=500
    training_args.weight_decay=0.01
    training_args.load_best_model_at_end=True
    training_args.metric_for_best_model="roc_auc"
    training_args.run_name=run_name
    training_args.warmup_steps=1
    training_args.gradient_accumulation_steps=5
    training_args.evaluate_during_training = True
    training_args.do_eval=True
    training_args.seed=11


    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)
    #get the label name
    label_name = data_args.task_name + "_label"

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None and data_args.train_file.endswith(".csv"):
        # Loading a dataset from local csv files
        datasets = load_dataset(
            "csv", data_files={"train": data_args.train_file, "validation": data_args.validation_file,
                               "test": data_args.test_file}
        )
        datasets = datasets.filter(lambda example: example[label_name] != "-1")
        logger.info(print(datasets['train']))
    else:
        # Loading a dataset from local json files
        datasets = load_dataset(
            "json", data_files={"train": data_args.train_file, "validation": data_args.validation_file}
        )
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        if multilabel and data_args.label_file is not None:
            with open(data_args.label_file) as code_file:
                label_list = code_file.read().split(" ")
                label_list.sort()
                num_labels = len(label_list)
        if not multilabel:
            label_list = datasets["train"].unique(label_name)
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
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

    # Setting the condition to train only the Adapter

    # Preprocessing the datasets
    if data_args.task_name is not None:
        text_key = task_to_keys[data_args.task_name]
        is_regression = data_args.task_name == "stsb"

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
        max_length = data_args.max_seq_length
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False
        max_length = None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None

    if data_args.task_name is not None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples['text'],)
        )
        result = tokenizer(*args, padding=padding, max_length=max_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)

        if label_name in examples:
            if multilabel:
                result["label"]=[]
                for e in examples[label_name]:
                    label_ids = [0] * len(label_list)
                    for l in e.split(","):
                        if l != "":
                            label_ids[label_list.index(l)] = 1
                    result["label"].append(label_ids)

            else:
                result["label"] = [label_to_id[l] for l in examples[label_name]]
        return result

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)

    train_dataset = datasets["train"]
    eval_dataset = datasets["validation"]
    test_dataset = datasets["test"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function

    # metric = load_metric("roc_auc", data_args.task_name)

    def softmax(vector):
        e = exp(vector)
        return e / e.sum()

    def compute_metrics(p: EvalPrediction):
        if multilabel:
            dim_size = len(p.label_ids[0])
            mask = np.ones((dim_size), dtype=bool)
            for c in range(dim_size):
                if (max(p.label_ids[:, c]) == 0):
                    mask[c] = False
            labels = p.label_ids[:, mask]
            y_score = np.array(p.predictions)[:, mask]
            filtered_cols = np.count_nonzero(mask == False)
            logger.info(f"{filtered_cols} columns not considered for ROC AUC calculation!")
            return {"roc_auc": roc_auc_score(y_true=labels, y_score=y_score,average="macro")}
        else:
            probs = [softmax(vector) for vector in p.predictions]
            return {"roc_auc": roc_auc_score(y_true=p.label_ids, y_score=probs, multi_class="ovo", average="macro")}

    tune_session=False
    # Initialize our Trainer
    def get_model():
        model = AutoModelWithHeads.from_pretrained(model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
        if tune_session:
            trial_id=tune.session.get_trial_id()
        model.add_classification_head(
        trial_id if tune_session else data_args.task_name,
        num_labels=num_labels,
        id2label={i: v for i, v in enumerate(label_list)} if num_labels > 0 and not multilabel else None ,
        overwrite_ok=True,
        multilabel=True if multilabel else False
        )
        adapter_args.train_adapter = True
        if adapter_args.train_adapter:
            adapter_config = AdapterConfig.load(
                adapter_args.adapter_config,
                non_linearity=adapter_args.adapter_non_linearity,
                reduction_factor=adapter_args.adapter_reduction_factor,
            )
            model.add_adapter(trial_id if tune_session else data_args.task_name, AdapterType.text_task, config=adapter_config)
             # Freeze all model weights except of those of this adapter
            model.train_adapter([trial_id if tune_session else data_args.task_name])
            # Set the adapters to be used in every forward pass
            model.set_active_adapters([trial_id if tune_session else data_args.task_name])
        return model


    trainer = ExtendedTrainer(
        model_init=get_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
        data_collator=default_data_collator if data_args.pad_to_max_length else None,
        do_save_full_model=False,
        do_save_adapters=True,
        callbacks=[DefaultFlowCallback,EarlyStoppingCallback(early_stopping_patience=10,greater_is_better=True)]
    )
    trainer.callback_handler.remove_callback(MLflowCallback)
    # Training
    if do_hpo:
        #ray.init(local_mode=True)
        # if ("RAY_HEAD_SERVICE_HOST" not in os.environ or os.environ["RAY_HEAD_SERVICE_HOST"] == ""):
        #     raise ValueError("RAY_HEAD_SERVICE_HOST environment variable empty."
        #                     "Is there a ray cluster running?")
        # os.environ['TUNE_DISABLE_AUTO_CALLBACK_LOGGERS'] = '1'
        # ray.init(address=os.environ["RAY_HEAD_SERVICE_HOST"] + ":7001")
        ray.init()
        tune_session=True
        logger.info("*** Hyperparameter Optimization Process Starting! ***")
        space = {
            "learning_rate": hp.uniform("learning_rate", 1e-6, 1e-4),
            "warmup_steps": hp.quniform("warmup_steps", 50, 1500, 1),
            "gradient_accumulation_steps": hp.quniform("gradient_accumulation_steps", 1, 20, 1)
        }
        scheduler = ASHAScheduler(
            metric="eval_roc_auc",
            mode="max",
            brackets=args.hpo_hyperband_brackets,
            grace_period=args.hpo_min_steps,
            max_t=args.hpo_max_steps,
            reduction_factor=3,
        )

        defaults = [{
            "learning_rate": 5e-5,
            "warmup_steps": 1000,
            "gradient_accumulation_steps": 1
        }]
        search = HyperOptSearch(
            space,
            metric="eval_roc_auc",
            mode="max",
            points_to_evaluate=defaults,
            n_initial_points=args.hpo_hp_initial_points
        )
        reporter = CLIReporter(
            parameter_columns={
                "learning_rate": "learning_rate",
                "gradient_accumulation_steps": "gradient_accumulation_steps",
                "warmup_steps": "warmup_steps"
            },
            metric_columns=[
                "eval_roc_auc", "eval_loss", "epoch", "training_iteration"
            ])

        def compute_objective(metrics):
            return metrics["eval_roc_auc"]

        best_trial=trainer.hyperparameter_search(direction="maximize",
                                                 backend="ray",
                                                 hp_space=lambda _: space,
                                                 scheduler=scheduler,
                                                 n_trials=args.hpo_num_samples,  # number of hyperparameter samples
                                                 # Aggressive termination of trials
                                                 keep_checkpoints_num=1,
                                                 checkpoint_score_attr="training_iteration",
                                                 search_alg=search,
                                                 fail_fast=True,
                                                 compute_objective=compute_objective,
                                                 resources_per_trial={
                                                     "cpu": 4,
                                                     "gpu": 1
                                                 },
                                                 local_dir="/data_dir/MTL/experiments/Adapters/dia_log/",
                                                 log_to_file=True,
                                                 progress_reporter=reporter,
                                                 )
        print(best_trial)

def arg_parse(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--hpo_num_samples", default=20, type=int, help="number of HPO trials to do")
    parser.add_argument("--hpo_min_steps", default=30, type=int,
                        help="number of HPO trial steps to do at minimum")  # Reduce to 20
    parser.add_argument("--hpo_max_steps", default=60, type=int,
                        help="number of HPO trial steps to do at maximum")  # Reduce to 85
    parser.add_argument("--hpo_hp_initial_points", default=1, type=int,
                        help="how many random trials to do before starting proper hpo")
    parser.add_argument("--hpo_hyperband_brackets", default=1, type=int, help="how many hyperband brackets to run")
    return parser.parse_args()

if __name__ == "__main__":
    parser = ArgumentParser()
    args = arg_parse(parser)
    doc_classification(args,config_file='/data_dir/MTL/experiments/Adapters/config_dia.json',run_name='dia_adap_1',do_hpo=True,multilabel=True)
