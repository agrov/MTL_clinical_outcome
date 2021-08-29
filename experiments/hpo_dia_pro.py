import ray
import os
import argparse
from argparse import Namespace, ArgumentParser
from multitask_doc_classification_dia_pro import doc_classification
from hyperopt import hp
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
import fire


def run_wrapper(config, *args, **kwargs):
    params=config["run_args"]
    params.lr = config["lr"]
    params.warmup_steps = config["warmup_steps"]
    params.embeds_dropout = config["embeds_dropout"]
    params.grad_acc_steps = config["grad_acc_steps"]
    #params.balance_classes = config["balance_classes"]
    #params.dia_plus = config["dia_plus"]
    #params.pro_plus = config["pro_plus"]
    params.task_config='/data_dir/MTL/experiments/configs/dia_pro_config.yaml'
    params.run_name='dia_pro_3'
    params.model_name_or_path='dmis-lab/biobert-v1.1'
    params.cache_dir='/data_dir/MTL/tasks/cache_dia_los'
    doc_classification(params)


def run_hpo(args):
    space = {
        "lr": hp.uniform("lr", 5e-6, 5e-4),
        "warmup_steps": hp.quniform("warmup_steps", 50, 1500, 1),
        "grad_acc_steps": hp.quniform("grad_acc_steps", 1, 20, 1),
        "embeds_dropout": hp.uniform("embeds_dropout", 0.1, 0.3)
        #"balance_classes":hp.choice("balance_classes", [True, False]),
        #"dia_plus": hp.uniform("dia_plus", 0.0001, 1),
        #"pro_plus": hp.uniform("pro_plus", 0.0001, 1)
    }
    defaults = [{
        "lr": 5e-5,
        "warmup_steps": 1000,
        "grad_acc_steps":1,
        "embeds_dropout": 0.1
        #"balance_classes": False,
        #"dia_plus": 1,
        #"pro_plus": 1
    }]
    search = HyperOptSearch(
        space,
        metric="roc_auc_dev",
        mode="max",
        points_to_evaluate=defaults,
        n_initial_points=args.hpo_hp_initial_points
    )
    scheduler = ASHAScheduler(
        metric="roc_auc_dev",
        mode="max",
        brackets=args.hpo_hyperband_brackets,
        grace_period=args.hpo_min_steps,
        max_t=args.hpo_max_steps,
        reduction_factor=3,
    )

    config = {
        "num_samples": args.hpo_num_samples,
        #"resources_per_trial": {"cpu": 1},
        "config": {
            "run_args": args
        }
    }
    analysis = ray.tune.run(
        run_wrapper,
        search_alg=search,
        local_dir='/home/anjali/MTL/experiments/logs_tensor/dia_pro/',
        scheduler=scheduler,
        fail_fast=True,
        **config
    )

    print("best config: ", analysis.get_best_config(metric="roc_auc_dev",mode="max"))
    print("best trial: ", analysis.get_best_trial(metric="roc_auc_dev", mode="max"))


def arg_parse(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--hpo_num_samples", default=1, type=int, help="number of HPO trials to do")
    parser.add_argument("--hpo_min_steps", default=2, type=int, help="number of HPO trial steps to do at minimum")#Reduce to 20
    parser.add_argument("--hpo_max_steps", default=2, type=int, help="number of HPO trial steps to do at maximum")#Reduce to 85
    parser.add_argument("--hpo_hp_initial_points", default=1, type=int,
                        help="how many random trials to do before starting proper hpo")
    parser.add_argument("--hpo_hyperband_brackets", default=1, type=int, help="how many hyperband brackets to run")
    return parser.parse_args()

if __name__ == "__main__":
    parser = ArgumentParser()
    args = arg_parse(parser)

    # if ("RAY_HEAD_SERVICE_HOST" not in os.environ
    #     or os.environ["RAY_HEAD_SERVICE_HOST"] == ""):
    #     raise ValueError("RAY_HEAD_SERVICE_HOST environment variable empty."
    #                        "Is there a ray cluster running?")
    # os.environ['TUNE_DISABLE_AUTO_CALLBACK_LOGGERS'] = '1'
    # #ray.init(address=os.environ["RAY_HEAD_SERVICE_HOST"] + ":6379")
    ray.init(local_mode=True)
    run_hpo(args)
