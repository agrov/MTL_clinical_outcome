import os
from pathlib import Path

from farm.modeling.language_model import LanguageModel
from transformers import BertConfig
from transformers.modeling_longformer import LongformerSelfAttention
from transformers.modeling_bert import BertModel
import torch
import logging

logger = logging.getLogger(__name__)


class BertLongLanguageModel(LanguageModel):
    @classmethod
    def load(cls, pretrained_model_name_or_path, language=None, **kwargs):
        """
        Load a pretrained model by supplying

        * the name of a remote model on s3 ("bert-base-cased" ...)
        * OR a local path of a model trained via transformers ("some_dir/huggingface_model")
        * OR a local path of a model trained via FARM ("some_dir/farm_model")

        :param pretrained_model_name_or_path: The path of the saved pretrained model or its name.
        :type pretrained_model_name_or_path: str

        """

        bert = cls()
        if "farm_lm_name" in kwargs:
            bert.name = kwargs["farm_lm_name"]
        else:
            bert.name = pretrained_model_name_or_path
        # We need to differentiate between loading model using FARM format and Pytorch-Transformers format
        farm_lm_config = Path(pretrained_model_name_or_path) / "language_model_config.json"
        if os.path.exists(farm_lm_config):
            # FARM style
            bert_config = BertConfig.from_pretrained(farm_lm_config)
            farm_lm_model = Path(pretrained_model_name_or_path) / "language_model.bin"
            bert.model = BertLong.from_pretrained(farm_lm_model, config=bert_config, **kwargs)
            bert.language = bert.model.config.language
        else:
            # Pytorch-transformer Style
            bert.model = BertLong.from_pretrained(str(pretrained_model_name_or_path), **kwargs)
            bert.language = cls._get_or_infer_language_from_name(language, pretrained_model_name_or_path)
        return bert

    def forward(
            self,
            input_ids,
            segment_ids,
            padding_mask,
            **kwargs,
    ):
        """
        Perform the forward pass of the BERT model.

        :param input_ids: The ids of each token in the input sequence. Is a tensor of shape [batch_size, max_seq_len]
        :type input_ids: torch.Tensor
        :param segment_ids: The id of the segment. For example, in next sentence prediction, the tokens in the
           first sentence are marked with 0 and those in the second are marked with 1.
           It is a tensor of shape [batch_size, max_seq_len]
        :type segment_ids: torch.Tensor
        :param padding_mask: A mask that assigns a 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size, max_seq_len]
        :return: Embeddings for each token in the input sequence.

        """
        padding_mask[:, [0]] = 2

        output_tuple = self.model(
            input_ids,
            token_type_ids=segment_ids,
            attention_mask=padding_mask,
        )
        if self.model.encoder.config.output_hidden_states == True:
            sequence_output, pooled_output, all_hidden_states = output_tuple[0], output_tuple[1], output_tuple[2]
            return sequence_output, pooled_output, all_hidden_states
        else:
            sequence_output, pooled_output = output_tuple[0], output_tuple[1]
            return sequence_output, pooled_output


class BertLongSelfAttention(LongformerSelfAttention):
    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
    ):
        return super().forward(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)


class BertLong(BertModel):
    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.encoder.layer):
            layer.attention.self = BertLongSelfAttention(config, layer_id=i)
