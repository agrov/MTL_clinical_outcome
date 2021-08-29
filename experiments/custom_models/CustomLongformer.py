import math
import os

import torch
from farm.modeling.prediction_head import TextClassificationHead
from torch import nn
import torch.nn.functional as F
from longformer.diagonaled_mm_tvm import diagonaled_mm as diagonaled_mm_tvm, mask_invalid_locations
from transformers.modeling_roberta import RobertaModel


class CustomLongformer(RobertaModel):
    def __init__(self, config):
        super(CustomLongformer, self).__init__(config)
        if config.attention_mode == 'n2':
            pass  # do nothing, use BertSelfAttention instead
        else:
            for i, layer in enumerate(self.encoder.layer):
                layer.attention.self = LongformerSelfAttention(config, layer_id=i)


class LongformerSelfAttention(nn.Module):
    def __init__(self, config, layer_id):
        super(LongformerSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size

        self.query = nn.Linear(config.hidden_size, self.embed_dim)
        self.key = nn.Linear(config.hidden_size, self.embed_dim)
        self.value = nn.Linear(config.hidden_size, self.embed_dim)

        self.query_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.key_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.value_global = nn.Linear(config.hidden_size, self.embed_dim)

        self.dropout = config.attention_probs_dropout_prob

        self.layer_id = layer_id
        self.attention_window = config.attention_window[self.layer_id]
        self.attention_dilation = config.attention_dilation[self.layer_id]
        assert self.attention_window > 0
        assert self.attention_dilation > 0
        self.autoregressive = config.autoregressive

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None):
        '''
        The `attention_mask` is changed in BertModel.forward from 0, 1, 2 to
            -ve: no attention
              0: local attention
            +ve: global attention
        '''
        if attention_mask is not None:
            attention_mask = attention_mask.squeeze(dim=2).squeeze(dim=1)
            key_padding_mask = attention_mask < 0
            extra_attention_mask = attention_mask > 0
            remove_from_windowed_attention_mask = attention_mask != 0

            num_extra_indices_per_batch = extra_attention_mask.long().sum(dim=1)
            max_num_extra_indices_per_batch = num_extra_indices_per_batch.max()
            has_same_length_extra_indices = (num_extra_indices_per_batch == max_num_extra_indices_per_batch).all()
        hidden_states = hidden_states.transpose(0, 1)
        seq_len, bsz, embed_dim = hidden_states.size()
        assert embed_dim == self.embed_dim
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        q /= math.sqrt(self.head_dim)

        q = q.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1).contiguous().float()
        k = k.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1).contiguous().float()
        # attn_weights = (bsz, seq_len, num_heads, window*2+1)
        attn_weights = diagonaled_mm_tvm(q, k, self.attention_window, self.attention_dilation, False, 0, False)
        mask_invalid_locations(attn_weights, self.attention_window, self.attention_dilation, False)
        if remove_from_windowed_attention_mask is not None:
            # This implementation is fast and takes very little memory because num_heads x hidden_size = 1
            # from (bsz x seq_len) to (bsz x seq_len x num_heads x hidden_size)
            remove_from_windowed_attention_mask = remove_from_windowed_attention_mask.unsqueeze(dim=-1).unsqueeze(
                dim=-1)
            # cast to float/half then replace 1's with -inf
            float_mask = remove_from_windowed_attention_mask.type_as(q).masked_fill(remove_from_windowed_attention_mask,
                                                                                    -10000.0)
            repeat_size = 1 if isinstance(self.attention_dilation, int) else len(self.attention_dilation)
            float_mask = float_mask.repeat(1, 1, repeat_size, 1)
            ones = float_mask.new_ones(size=float_mask.size())  # tensor of ones
            # diagonal mask with zeros everywhere and -inf inplace of padding
            d_mask = diagonaled_mm_tvm(ones, float_mask, self.attention_window, self.attention_dilation, False, 0,
                                       False)
            attn_weights += d_mask
        assert list(attn_weights.size()) == [bsz, seq_len, self.num_heads, self.attention_window * 2 + 1]

        # the extra attention
        if extra_attention_mask is not None:
            if has_same_length_extra_indices:
                # a simplier implementation for efficiency
                # k = (bsz, seq_len, num_heads, head_dim)
                selected_k = k.masked_select(extra_attention_mask.unsqueeze(-1).unsqueeze(-1)).view(bsz,
                                                                                                    max_num_extra_indices_per_batch,
                                                                                                    self.num_heads,
                                                                                                    self.head_dim)
                # selected_k = (bsz, extra_attention_count, num_heads, head_dim)
                # selected_attn_weights = (bsz, seq_len, num_heads, extra_attention_count)
                selected_attn_weights = torch.einsum('blhd,bshd->blhs', (q, selected_k))
            else:
                # since the number of extra attention indices varies across
                # the batch, we need to process each element of the batch
                # individually
                flat_selected_k = k.masked_select(extra_attention_mask.unsqueeze(-1).unsqueeze(-1))
                selected_attn_weights = torch.ones(
                    bsz, seq_len, self.num_heads, max_num_extra_indices_per_batch, device=k.device, dtype=k.dtype
                )
                selected_attn_weights.fill_(-10000.0)
                start = 0
                for i in range(bsz):
                    end = start + num_extra_indices_per_batch[i] * self.num_heads * self.head_dim
                    # the selected entries for this batch element
                    i_selected_k = flat_selected_k[start:end].view(-1, self.num_heads, self.head_dim)
                    # (seq_len, num_heads, num extra indices)
                    i_selected_attn_weights = torch.einsum('lhd,shd->lhs', (q[i, :, :, :], i_selected_k))
                    selected_attn_weights[i, :, :, :num_extra_indices_per_batch[i]] = i_selected_attn_weights
                    start = end

            # concat to attn_weights
            # (bsz, seq_len, num_heads, extra attention count + 2*window+1)
            attn_weights = torch.cat((selected_attn_weights, attn_weights), dim=-1)

        attn_weights_float = F.softmax(attn_weights, dim=-1)
        if key_padding_mask is not None:
            # softmax sometimes inserts NaN if all positions are masked, replace them with 0
            attn_weights_float = torch.masked_fill(attn_weights_float, key_padding_mask.unsqueeze(-1).unsqueeze(-1),
                                                   0.0)

        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)
        v = v.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1).contiguous().float()
        attn = 0
        if extra_attention_mask is not None and max_num_extra_indices_per_batch > 0:
            selected_attn_probs = attn_probs.narrow(-1, 0, max_num_extra_indices_per_batch)
            if has_same_length_extra_indices:
                selected_v = v.masked_select(
                    extra_attention_mask.unsqueeze(-1).unsqueeze(-1)
                ).view(bsz, max_num_extra_indices_per_batch, self.num_heads, self.head_dim)
            else:
                flat_selected_v = v.masked_select(extra_attention_mask.unsqueeze(-1).unsqueeze(-1))
                # don't worry about masking since this is multiplied by attn_probs, and masking above
                # before softmax will remove masked entries
                selected_v = torch.zeros(
                    bsz, max_num_extra_indices_per_batch, self.num_heads, self.head_dim, device=v.device, dtype=v.dtype
                )
                start = 0
                for i in range(bsz):
                    end = start + num_extra_indices_per_batch[i] * self.num_heads * self.head_dim
                    i_selected_v = flat_selected_v[start:end].view(-1, self.num_heads, self.head_dim)
                    selected_v[i, :num_extra_indices_per_batch[i], :, :] = i_selected_v
                    start = end
            attn = torch.einsum('blhs,bshd->blhd', (selected_attn_probs, selected_v))
            attn_probs = attn_probs.narrow(-1, max_num_extra_indices_per_batch,
                                           attn_probs.size(-1) - max_num_extra_indices_per_batch).contiguous()

        attn += diagonaled_mm_tvm(attn_probs, v, self.attention_window, self.attention_dilation, True, 0, False)
        attn = attn.type_as(hidden_states)
        assert list(attn.size()) == [bsz, seq_len, self.num_heads, self.head_dim]
        attn = attn.transpose(0, 1).reshape(seq_len, bsz, embed_dim).contiguous()

        # For this case, we'll just recompute the attention for these indices
        # and overwrite the attn tensor. TODO: remove the redundant computation
        if extra_attention_mask is not None and max_num_extra_indices_per_batch > 0:
            if has_same_length_extra_indices:
                # query = (seq_len, bsz, dim)
                # extra_attention_mask = (bsz, seq_len)
                # selected_query = (max_num_extra_indices_per_batch, bsz, embed_dim)
                selected_hidden_states = hidden_states.masked_select(
                    extra_attention_mask.transpose(0, 1).unsqueeze(-1)).view(max_num_extra_indices_per_batch, bsz,
                                                                             embed_dim)
                # if *_proj_full exists use them, otherwise default to *_proj
                q = self.query_global(selected_hidden_states)
                k = self.key_global(hidden_states)
                v = self.value_global(hidden_states)
                q /= math.sqrt(self.head_dim)

                q = q.contiguous().view(max_num_extra_indices_per_batch, bsz * self.num_heads, self.head_dim).transpose(
                    0, 1)  # (bsz*self.num_heads, max_num_extra_indices_per_batch, head_dim)
                k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0,
                                                                                           1)  # bsz * self.num_heads, seq_len, head_dim)
                v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0,
                                                                                           1)  # bsz * self.num_heads, seq_len, head_dim)
                attn_weights = torch.bmm(q, k.transpose(1, 2))
                assert list(attn_weights.size()) == [bsz * self.num_heads, max_num_extra_indices_per_batch, seq_len]
                if key_padding_mask is not None:
                    attn_weights = attn_weights.view(bsz, self.num_heads, max_num_extra_indices_per_batch, seq_len)
                    attn_weights = attn_weights.masked_fill(
                        key_padding_mask.unsqueeze(1).unsqueeze(2),
                        float('-inf'),
                    )
                    attn_weights = attn_weights.view(bsz * self.num_heads, max_num_extra_indices_per_batch, seq_len)
                attn_weights_float = F.softmax(attn_weights, dim=-1)
                attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)
                selected_attn = torch.bmm(attn_probs, v)
                assert list(selected_attn.size()) == [bsz * self.num_heads, max_num_extra_indices_per_batch,
                                                      self.head_dim]
                selected_attn = selected_attn.transpose(0, 1).contiguous().view(
                    max_num_extra_indices_per_batch * bsz * embed_dim)

                # now update attn by filling in the relevant indices with selected_attn
                # masked_fill_ only allows floats as values so this doesn't work
                # attn.masked_fill_(extra_attention_mask.transpose(0, 1).unsqueeze(-1), selected_attn)
                attn[extra_attention_mask.transpose(0, 1).unsqueeze(-1).repeat((1, 1, embed_dim))] = selected_attn
            else:
                raise ValueError  # not implemented

        context_layer = attn.transpose(0, 1)
        if self.output_attentions:
            if extra_attention_mask is not None and max_num_extra_indices_per_batch > 0:
                # With global attention, return global attention probabilities only
                # batch_size x num_heads x num_global_attention_tokens x sequence_length
                # which is the attention weights from tokens with global attention to all tokens
                # It doesn't not return local attention
                attn_weights = attn_weights.view(bsz, self.num_heads, max_num_extra_indices_per_batch, seq_len)
            else:
                # without global attention, return local attention probabilities
                # batch_size x num_heads x sequence_length x window_size
                # which is the attention weights of every token attending to its neighbours
                attn_weights = attn_weights.permute(0, 2, 1, 3)
        outputs = (context_layer, attn_weights) if self.output_attentions else (context_layer,)
        return outputs


class LongformerNextSentenceHead(TextClassificationHead):
    """
    Almost identical to a TextClassificationHead. Only difference: we can load the weights from
     a pretrained language model that was saved in the Transformers style (all in one model).
    """

    @classmethod
    def load(cls, pretrained_model_name_or_path):
        """
        Load a prediction head from a saved FARM or transformers model. `pretrained_model_name_or_path`
        can be one of the following:
        a) Local path to a FARM prediction head config (e.g. my-bert/prediction_head_0_config.json)
        b) Local path to a Transformers model (e.g. my-bert)
        c) Name of a public model from https://huggingface.co/models (e.g.bert-base-cased)


        :param pretrained_model_name_or_path: local path of a saved model or name of a publicly available model.
                                              Exemplary public names:
                                              - bert-base-cased

                                              See https://huggingface.co/models for full list

        """
        if os.path.exists(pretrained_model_name_or_path) \
                and "config.json" in pretrained_model_name_or_path \
                and "prediction_head" in pretrained_model_name_or_path:
            # a) FARM style
            head = super(LongformerNextSentenceHead, cls).load(pretrained_model_name_or_path)
        else:
            # b) pytorch-transformers style
            # load weights from longformer model
            # (we might change this later to load directly from a state_dict to generalize for other language models)
            bert_with_lm = BertForPreTraining.from_pretrained(pretrained_model_name_or_path)

            # init empty head
            head = cls(layer_dims=[bert_with_lm.config.hidden_size, 2], loss_ignore_index=-1, task_name="nextsentence")

            # load weights
            head.feed_forward.feed_forward[0].load_state_dict(bert_with_lm.cls.seq_relationship.state_dict())
            del bert_with_lm

        return head

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

        return head


class LongformerForPreTraining(RobertaModel):
    r"""
        **masked_lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        **next_sentence_label**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair (see ``input_ids`` docstring)
            Indices should be in ``[0, 1]``.
            ``0`` indicates sequence B is a continuation of sequence A,
            ``1`` indicates sequence B is a random sequence.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when both ``masked_lm_labels`` and ``next_sentence_label`` are provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total loss as the sum of the masked language modeling loss and the next sequence prediction (classification) loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **seq_relationship_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, 2)``
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForPreTraining.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        prediction_scores, seq_relationship_scores = outputs[:2]

    """

    def __init__(self, config):
        super(LongformerForPreTraining, self).__init__(config)

        self.longformer = CustomLongformer(config)
        self.cls = BertPreTrainingHeads(config)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.cls.predictions.decoder,
                                   self.bert.embeddings.word_embeddings)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                masked_lm_labels=None, next_sentence_label=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        outputs = (prediction_scores, seq_relationship_score,) + outputs[
                                                                 2:]  # add hidden states and attention if they are here

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            outputs = (total_loss,) + outputs

        return outputs  # (loss), prediction_scores, seq_relationship_score, (hidden_states), (attentions)
