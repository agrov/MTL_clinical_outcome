import logging

from farm.modeling.prediction_head import PredictionHead
logger = logging.getLogger(__name__)


class CosineSimilarityHead(PredictionHead):
    def __init__(
            self,
            layer_dims=None,
            num_labels=None,
            class_weights=None,
            loss_ignore_index=-100,
            loss_reduction="none",
            task_name="vector_similarity",
            **kwargs,
    ):
        """
        :param layer_dims: The size of the layers in the feed forward component. The feed forward will have as many layers as there are ints in this list. This param will be deprecated in future
        :type layer_dims: list
        :param num_labels: The numbers of labels. Use to set the size of the final layer in the feed forward component. It is recommended to only set num_labels or layer_dims, not both.
        :type num_labels: int
        :param class_weights:
        :param loss_ignore_index:
        :param loss_reduction:
        :param task_name:
        :param kwargs:
        """
        super(CosineSimilarityHead, self).__init__()
        # num_labels could in most cases also be automatically retrieved from the data processor
        if layer_dims:
            self.layer_dims = layer_dims
            logger.warning("`layer_dims` will be deprecated in future releases")
        elif num_labels:
            self.layer_dims = [768, num_labels]
        else:
            raise ValueError("Please supply `num_labels` to define output dim of prediction head")
        self.num_labels = self.layer_dims[-1]
        self.feed_forward = FeedForwardBlock(self.layer_dims)
        logger.info(f"Prediction head initialized with size {self.layer_dims}")
        self.num_labels = self.layer_dims[-1]
        self.ph_output_type = "per_sequence"
        self.model_type = "text_classification"
        self.task_name = task_name  # used for connecting with the right output of the processor
        self.class_weights = class_weights

        if class_weights:
            logger.info(f"Using class weights for task '{self.task_name}': {self.class_weights}")
            balanced_weights = nn.Parameter(torch.tensor(class_weights), requires_grad=False)
        else:
            balanced_weights = None

        self.loss_fct = CrossEntropyLoss(
            weight=balanced_weights,
            reduction=loss_reduction,
            ignore_index=loss_ignore_index,
        )

        self.generate_config()

    @classmethod
    def load(cls, pretrained_model_name_or_path):
        """
        Load a prediction head from a saved FARM or transformers model. `pretrained_model_name_or_path`
        can be one of the following:
        a) Local path to a FARM prediction head config (e.g. my-bert/prediction_head_0_config.json)
        b) Local path to a Transformers model (e.g. my-bert)
        c) Name of a public model from https://huggingface.co/models (e.g. distilbert-base-uncased-distilled-squad)


        :param pretrained_model_name_or_path: local path of a saved model or name of a publicly available model.
                                              Exemplary public name:
                                              - deepset/bert-base-german-cased-hatespeech-GermEval18Coarse

                                              See https://huggingface.co/models for full list

        """

        if os.path.exists(pretrained_model_name_or_path) \
                and "config.json" in pretrained_model_name_or_path \
                and "prediction_head" in pretrained_model_name_or_path:
            # a) FARM style
            head = super(TextClassificationHead, cls).load(pretrained_model_name_or_path)
        else:
            # b) transformers style
            # load all weights from model
            full_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path)
            # init empty head
            head = cls(layer_dims=[full_model.config.hidden_size, len(full_model.config.id2label)])
            # transfer weights for head from full model
            head.feed_forward.feed_forward[0].load_state_dict(full_model.classifier.state_dict())
            del full_model

        cls.head = head
        return cls.head

    def forward(self, X):
        logits = self.feed_forward(X)
        return logits

    def logits_to_loss(self, logits, **kwargs):
        label_ids = kwargs.get(self.label_tensor_name)
        return self.loss_fct(logits, label_ids.view(-1))

    def logits_to_probs(self, logits, return_class_probs=False, **kwargs):
        softmax = torch.nn.Softmax(dim=1)
        probs = softmax(logits)
        if return_class_probs:
            probs = probs.cpu().numpy()
        else:
            pred_ids = logits.argmax(1)
            probs = torch.max(probs, dim=1)[0]
            probs = probs.cpu().numpy()
            # Turn probabilities of zero class into 1-prob values
            # Warning! Currently only works with binary classification
            probs = [val if pred_ids[i] == 1 else 1 - val for i, val in enumerate(probs)]
        return probs

    def logits_to_preds(self, logits, **kwargs):
        logits = logits.cpu().numpy()
        pred_ids = logits.argmax(1)
        preds = [self.label_list[int(x)] for x in pred_ids]
        return preds

    def prepare_labels(self, **kwargs):
        label_ids = kwargs.get(self.label_tensor_name)
        label_ids = label_ids.cpu().numpy()
        labels = [self.label_list[int(x)] for x in label_ids]
        return labels

    def formatted_preds(self, logits, samples, return_class_probs=False, **kwargs):
        preds = self.logits_to_preds(logits)
        probs = self.logits_to_probs(logits, return_class_probs)
        contexts = [sample.clear_text["text"] for sample in samples]
        contexts_b = [sample.clear_text["text_b"] for sample in samples if "text_b" in sample.clear_text]
        if len(contexts_b) != 0:
            contexts = ["|".join([a, b]) for a, b in zip(contexts, contexts_b)]

        assert len(preds) == len(probs) == len(contexts)

        res = {"task": "text_classification", "predictions": []}
        for pred, prob, context in zip(preds, probs, contexts):
            if not return_class_probs:
                pred_dict = {
                    "start": None,
                    "end": None,
                    "context": f"{context}",
                    "label": f"{pred}",
                    "probability": prob,
                }
            else:
                pred_dict = {
                    "start": None,
                    "end": None,
                    "context": f"{context}",
                    "label": "class_probabilities",
                    "probability": prob,
                }

            res["predictions"].append(pred_dict)
        return res
