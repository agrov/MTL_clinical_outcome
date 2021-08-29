from pathlib import Path

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import TextClassificationProcessor
from farm.eval import Evaluator
from farm.modeling.optimization import initialize_optimizer
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import MultiLabelTextClassificationHead
from farm.modeling.tokenization import Tokenizer
from farm.evaluation.metrics import register_metrics
from farm.train import EarlyStopping
from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings
from ray import tune
import yaml
import fire
import numpy as np

from custom_models.evaluation.ExtendedEvaluator import ExtendedEvaluator
from custom_models.training.ExtendedAdaptiveModel import ExtendedAdaptiveModel
from custom_models.training.ExtendedTrainer import ExtendedTrainer
from custom_models.training.TextClassificationHead import TextClassificationHead
from custom_models.Longformer import BertLongLanguageModel  # Import is required in order to load BertLongLanguageModel
import metrics
import utils
from argparse import Namespace, ArgumentParser
logger = utils.get_logger(__name__)


def doc_classification(args):
    task_config=args.task_config
    run_name=args.run_name
    model_name_or_path=args.model_name_or_path
    cache_dir=args.cache_dir
    lr=args.lr
    warmup_steps=args.warmup_steps
    balance_classes=args.balance_classes
    embeds_dropout=args.embeds_dropout
    grad_acc_steps=args.grad_acc_steps
    mp=args.mp
    dia_plus=1
    batch_size=20
    epochs=200
    early_stopping_metric="roc_auc"
    early_stopping_mode="none"
    early_stopping_patience=20
    model_class="Bert"
    tokenizer_class="BertTokenizer"
    do_lower_case=False
    do_train=True
    do_eval=True
    do_hpo=True
    print_preds=False
    print_dev_preds=False
    max_seq_len=512
    seed=11
    eval_every=500
    use_amp=False
    use_cuda=True
    task_weights = { "dia_plus":dia_plus,"mp" : mp}
    
    # Load task config
    task_config = yaml.safe_load(open(task_config))
    tasks = task_config["tasks"]

    data_dir = Path(task_config["data"]["data_dir"])
    save_dir = utils.init_save_dir(task_config["output_dir"],
                                   task_config["experiment_name"],
                                   run_name,
                                   tune.session.get_trial_name() if do_hpo else None)
    if do_hpo:
        trial_id=tune.session.get_trial_name()
    # Create label list from args list or (for large label lists) create from file by splitting by space
    for task in tasks:
        if isinstance(task["data"]["label_list"], str):
            with open(task["data"]["label_list"]) as code_file:
                task["data"]["label_list"] = code_file.read().split(" ")
                # add -1 label for cases where the task should not be solved,
                # e.g. length of stay prediction for patients who deceased during the hospital stay
                if "-1" not in task["data"]["label_list"]:
                    task["data"]["label_list"] = ["-1"] + task["data"]["label_list"]

        # Register Outcome Metrics per task
        if task["metric"].startswith("binary"):
            register_metrics(f"binary_classification_metrics_{task['name']}",
                             metrics.binary_classification_metrics)
        elif task["metric"].startswith("multiclass"):
            register_metrics(f"multiclass_classification_metrics_{task['name']}",
                             metrics.multiclass_classification_metrics)
        elif task["metric"].startswith("multilabel"):
            metrics.register_multilabel_classification_metrics_3_digits_only(
                f"multilabel_classification_metrics_3_digits_only_{task['name']}",
                task["data"]["label_list"])

    # General Settings
    set_all_seeds(seed=seed)
    device, n_gpu = initialize_device_settings(use_cuda=use_cuda, use_amp=use_amp)

    # 1.Create a tokenizer
    tokenizer = Tokenizer.load(pretrained_model_name_or_path=model_name_or_path, tokenizer_class=tokenizer_class,
                               do_lower_case=do_lower_case)

    # 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
    processor = TextClassificationProcessor(tokenizer=tokenizer,
                                            max_seq_len=max_seq_len,
                                            data_dir=data_dir,
                                            train_filename=task_config["data"]["train_filename"],
                                            dev_filename=task_config["data"]["dev_filename"],
                                            dev_split=task_config["data"]["dev_split"] if "dev_split" in task_config[
                                                "data"] else None,
                                            test_filename=task_config["data"]["test_filename"],
                                            delimiter=task_config["data"]["parsing"]["delimiter"],
                                            quote_char=task_config["data"]["parsing"]["quote_char"]
                                            )

    for task in tasks:
        task_name = task["name"]
        task_label_column = task["data"]["parsing"]["label_column"]
        task_type = "multilabel_classification" if task["multilabel"] else "classification"

        processor.add_task(name=task_name,
                           label_list=task["data"]["label_list"],
                           metric=task["metric"],
                           label_column_name=task_label_column,
                           label_name=f"{task_label_column}s",
                           task_type=task_type,
                           text_column_name="text")

        task["weight_factor"] = task_weights[task_name] if task_weights else 1
        
    
    # 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a
    #    few descriptive statistics of our datasets
    data_silo = DataSilo(
        processor=processor,
        caching=True,
        cache_path=Path(cache_dir),
        batch_size=batch_size)

    if do_train:

        # Setup MLFlow logger
        ml_logger = MLFlowLogger(tracking_uri=task_config["log_dir"])
        ml_logger.init_experiment(experiment_name=task_config["experiment_name"],
                                  run_name=f'{task_config["experiment_name"]}_{run_name}')

        # 4. Create an AdaptiveModel
        # a) which consists of a pretrained language model as a basis
        language_model = LanguageModel.load(model_name_or_path, language_model_class=model_class)

        # b) and a prediction head on top that is suited for our task

        prediction_heads = []
        output_types = []
        for task in tasks:

            class_weights = len(task["data"]["label_list"]) * [1]

            if balance_classes:
                class_weights = data_silo.calculate_class_weights(task_name=task["name"])

            if len(task["data"]["label_list"]) > 2:
                class_weights[0] = 0  # make sure that the -1 label at the beginning does not get weighted

            class_weights = np.array(class_weights) * task["weight_factor"]

            if task["multilabel"]:

                prediction_head = MultiLabelTextClassificationHead(
                    class_weights=class_weights,
                    task_name=task["name"],
                    num_labels=len(task["data"]["label_list"]))

            else:
                prediction_head = TextClassificationHead(
                    class_weights=class_weights,
                    task_name=task["name"],
                    num_labels=len(task["data"]["label_list"]))

            prediction_heads.append(prediction_head)
            output_types.append(task["output_type"])

        model = ExtendedAdaptiveModel(
            language_model=language_model,
            prediction_heads=prediction_heads,
            embeds_dropout_prob=embeds_dropout,
            lm_output_types=output_types,
            device=device)

        # 5. Create an optimizer
        schedule_opts = {"name": "LinearWarmup",
                         "num_warmup_steps": warmup_steps}

        model, optimizer, lr_schedule = initialize_optimizer(
            model=model,
            learning_rate=lr,
            device=device,
            n_batches=len(data_silo.loaders["train"]),
            n_epochs=epochs,
            use_amp=use_amp,
            grad_acc_steps=grad_acc_steps,
            schedule_opts=schedule_opts)

        # 6. Create an early stopping instance
        early_stopping = None
        if early_stopping_mode != "none":
            early_stopping = EarlyStopping(
                mode=early_stopping_mode,
                min_delta=0.0001,
                save_dir=save_dir,
                metric=early_stopping_metric,
                patience=early_stopping_patience
            )

        # 7. Feed everything to the Trainer, which keeps care of growing our model into powerful plant and evaluates it
        # from time to time

        trainer = ExtendedTrainer(
            model=model,
            optimizer=optimizer,
            data_silo=data_silo,
            epochs=epochs,
            n_gpu=n_gpu,
            lr_schedule=lr_schedule,
            evaluate_every=eval_every,
            early_stopping=early_stopping,
            device=device,
            grad_acc_steps=grad_acc_steps,
            evaluator_test=do_eval
        )

        def score_callback(eval_score, train_loss):
            tune.report(roc_auc_dev=eval_score, train_loss=train_loss)

        # 8. Train the model
        trainer.train(score_callback=score_callback if do_hpo else None)

        # 9. Save model if not saved in early stopping
        model.save(save_dir / "final_model")
        processor.save(save_dir / "final_model")

    if do_eval:
        # Load newly trained model or existing model
        if do_train:
            model_dir = save_dir
        else:
            model_dir = Path(model_name_or_path)

        logger.info("###### Eval on TEST SET #####")

        evaluator_test = ExtendedEvaluator(
            data_loader=data_silo.get_data_loader("test"),
            tasks=data_silo.processor.tasks,
            device=device
        )

        # Load trained model for evaluation
        model = ExtendedAdaptiveModel.load(model_dir, device)
        model.connect_heads_with_processor(data_silo.processor.tasks, require_labels=True)

        # Evaluate
        results,multi_roc_auc_test = evaluator_test.eval(model, return_preds_and_labels=True)

        # Log results
        utils.log_results(results, dataset_name="test", steps=len(evaluator_test.data_loader),
                          save_path=model_dir / "eval_results.txt")

        if print_preds:
            # Print model test predictions
            utils.save_predictions(results, save_dir=model_dir, multilabel=task_config["multilabel"])

        if print_dev_preds:
            # Evaluate on dev set, e.g. for threshold tuning
            evaluator_dev = Evaluator(
                data_loader=data_silo.get_data_loader("dev"),
                tasks=data_silo.processor.tasks,
                device=device
            )
            dev_results = evaluator_dev.eval(model, return_preds_and_labels=True)
            utils.log_results(dev_results, dataset_name="dev", steps=len(evaluator_dev.data_loader),
                              save_path=model_dir / "eval_dev_results.txt")

            # Print model dev predictions
            utils.save_predictions(dev_results, save_dir=model_dir, multilabel=task_config["multilabel"],
                                   dataset_name="dev")


if __name__ == '__main__':
    fire.Fire(doc_classification)
