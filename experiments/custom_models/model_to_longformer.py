import fire
from transformers import BertModel, LongformerSelfAttention, BertTokenizerFast
import torch
import copy
import logging

logger = logging.getLogger(__name__)


def bert_to_longformer(model_name_or_path, max_seq_len, save_dir, attention_window=512):
    model = BertModel.from_pretrained(model_name_or_path)
    tokenizer = BertTokenizerFast.from_pretrained(model_name_or_path, model_max_length=max_seq_len)
    config = model.config

    # extend position embeddings
    current_max_pos, embed_size = model.embeddings.position_embeddings.weight.shape

    config.max_position_embeddings = max_seq_len
    assert max_seq_len > current_max_pos
    # allocate a larger position embedding matrix
    new_pos_embed = model.embeddings.position_embeddings.weight.new_empty(max_seq_len, embed_size)
    # copy position embeddings over and over to initialize the new position embeddings
    k = 0
    step = current_max_pos
    while k < max_seq_len - 1:
        new_pos_embed[k:(k + step)] = model.embeddings.position_embeddings.weight
        k += step
    model.embeddings.position_embeddings.weight.data = new_pos_embed
    model.embeddings.position_ids.data = torch.tensor([i for i in range(max_seq_len)]).reshape(1, max_seq_len)

    # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
    config.attention_window = [attention_window] * config.num_hidden_layers
    for i, layer in enumerate(model.encoder.layer):
        longformer_self_attn = LongformerSelfAttention(config, layer_id=i)
        longformer_self_attn.query = layer.attention.self.query
        longformer_self_attn.key = layer.attention.self.key
        longformer_self_attn.value = layer.attention.self.value

        longformer_self_attn.query_global = copy.deepcopy(layer.attention.self.query)
        longformer_self_attn.key_global = copy.deepcopy(layer.attention.self.key)
        longformer_self_attn.value_global = copy.deepcopy(layer.attention.self.value)

        layer.attention.self = longformer_self_attn

    logger.info(f'saving model and tokenizer to {save_dir}')
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


def model_to_longformer(model_name_or_path, max_seq_len, save_dir, attention_window=512, base_model_class="Bert"):
    if base_model_class == "Bert":
        bert_to_longformer(model_name_or_path, max_seq_len, save_dir, attention_window)
    else:
        raise NotImplementedError(f"Model class '{base_model_class}' not yet implemented.")


if __name__ == '__main__':
    fire.Fire(model_to_longformer)
