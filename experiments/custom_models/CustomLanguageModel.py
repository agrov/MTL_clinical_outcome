from farm.modeling.language_model import LanguageModel, Roberta
import json
import logging
import os
from pathlib import Path

import torch
# from longformer.longformer import LongformerConfig

# from outcome_pretraining.CustomLongformer import CustomLongformer
from transformers import RobertaConfig, RobertaModel

logger = logging.getLogger(__name__)


class CustomLanguageModel(LanguageModel):

    def __init_subclass__(cls, **kwargs):
        """ This automatically keeps track of all available subclasses.
        Enables generic load() or all specific LanguageModel implementation.
        """
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    @classmethod
    def load(cls, pretrained_model_name_or_path, n_added_tokens=0, language_model_class=None, **kwargs):
        """
        Load a pretrained language model either by

        1. specifying its name and downloading it
        2. or pointing to the directory it is saved in.

        Available remote models:

        * bert-base-uncased
        * bert-large-uncased
        * bert-base-cased
        * bert-large-cased
        * bert-base-multilingual-uncased
        * bert-base-multilingual-cased
        * bert-base-chinese
        * bert-base-german-cased
        * roberta-base
        * roberta-large
        * xlnet-base-cased
        * xlnet-large-cased
        * xlm-roberta-base
        * xlm-roberta-large
        * albert-base-v2
        * albert-large-v2
        * distilbert-base-german-cased
        * distilbert-base-multilingual-cased

        See all supported model variations here: https://huggingface.co/models

        The appropriate language model class is inferred automatically from `pretrained_model_name_or_path`
        or can be manually supplied via `language_model_class`.

        :param pretrained_model_name_or_path: The path of the saved pretrained model or its name.
        :type pretrained_model_name_or_path: str
        :param language_model_class: (Optional) Name of the language model class to load (e.g. `Bert`)
        :type language_model_class: str

        """
        config_file = Path(pretrained_model_name_or_path) / "language_model_config.json"
        if os.path.exists(config_file):
            # it's a local directory in FARM format
            config = json.load(open(config_file))
            language_model = cls.subclasses[config["name"]].load(pretrained_model_name_or_path)
        else:
            if language_model_class is None:
                # it's transformers format (either from model hub or local)
                pretrained_model_name_or_path = str(pretrained_model_name_or_path)
                if "xlm" in pretrained_model_name_or_path and "roberta" in pretrained_model_name_or_path:
                    language_model_class = 'XLMRoberta'
                elif 'roberta' in pretrained_model_name_or_path:
                    language_model_class = 'CustomRoberta'
                elif 'albert' in pretrained_model_name_or_path:
                    language_model_class = 'Albert'
                elif 'distilbert' in pretrained_model_name_or_path:
                    language_model_class = 'DistilBert'
                elif 'bert' in pretrained_model_name_or_path:
                    language_model_class = 'Bert'
                elif 'xlnet' in pretrained_model_name_or_path:
                    language_model_class = 'XLNet'
                elif 'longformer' in pretrained_model_name_or_path:
                    language_model_class = 'Longformer'

            if language_model_class:
                language_model = cls.subclasses[language_model_class].load(pretrained_model_name_or_path, **kwargs)
            else:
                language_model = None

            if language_model_class == 'XLMRoberta':
                # TODO: for some reason, the pretrained XLMRoberta has different vocab size in the tokenizer compared to the model this is a hack to resolve that
                n_added_tokens = 3

        if not language_model:
            raise Exception(
                f"Model not found for {pretrained_model_name_or_path}. Either supply the local path for a saved "
                f"model or one of bert/roberta/xlnet/albert/distilbert models that can be downloaded from remote. "
                f"Ensure that the model class name can be inferred from the directory name when loading a "
                f"Transformers' model. Here's a list of available models: "
                f"https://farm.deepset.ai/api/modeling.html#farm.modeling.language_model.LanguageModel.load"
            )

        # resize embeddings in case of custom vocab
        if n_added_tokens != 0:
            # TODO verify for other models than BERT
            model_emb_size = language_model.model.resize_token_embeddings(new_num_tokens=None).num_embeddings
            vocab_size = model_emb_size + n_added_tokens
            logger.info(
                f"Resizing embedding layer of LM from {model_emb_size} to {vocab_size} to cope with custom vocab.")
            language_model.model.resize_token_embeddings(vocab_size)
            # verify
            model_emb_size = language_model.model.resize_token_embeddings(new_num_tokens=None).num_embeddings
            assert vocab_size == model_emb_size

        return language_model


class CustomRoberta(Roberta):

    @classmethod
    def load(cls, pretrained_model_name_or_path, language=None, **kwargs):
        """
        Load a language model either by supplying

        * the name of a remote model on s3 ("roberta-base" ...)
        * or a local path of a model trained via transformers ("some_dir/huggingface_model")
        * or a local path of a model trained via FARM ("some_dir/farm_model")

        :param pretrained_model_name_or_path: name or path of a model
        :param language: (Optional) Name of language the model was trained for (e.g. "german").
                         If not supplied, FARM will try to infer it from the model name.
        :return: Language Model

        """
        roberta = cls()
        if "farm_lm_name" in kwargs:
            roberta.name = kwargs["farm_lm_name"]
        else:
            roberta.name = pretrained_model_name_or_path
        # We need to differentiate between loading model using FARM format and Pytorch-Transformers format
        farm_lm_config = Path(pretrained_model_name_or_path) / "language_model_config.json"
        if os.path.exists(farm_lm_config):
            # FARM style
            config = RobertaConfig.from_pretrained(farm_lm_config)
            farm_lm_model = Path(pretrained_model_name_or_path) / "language_model.bin"
            roberta.model = RobertaModel.from_pretrained(farm_lm_model, config=config, **kwargs)
            roberta.language = roberta.model.config.language
        else:
            # Huggingface transformer Style
            roberta.model = RobertaModel.from_pretrained(str(pretrained_model_name_or_path), **kwargs)

            roberta.model.config.type_vocab_size = 2
            single_emb = roberta.model.embeddings.token_type_embeddings
            roberta.model.embeddings.token_type_embeddings = torch.nn.Embedding(2, single_emb.embedding_dim)
            roberta.model.embeddings.token_type_embeddings.weight = torch.nn.Parameter(single_emb.weight.repeat([2, 1]))

            roberta.language = cls._get_or_infer_language_from_name(language, pretrained_model_name_or_path)
        return roberta