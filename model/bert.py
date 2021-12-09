import torch
from torch import nn
from transformers import AutoModel

CHINESE_BERT_NAME = "bert-base-chinese"
ENGLISH_BERT_NAME = "bert-base-cased"
BERT_DIM = 768


class ChineseBERT(nn.Module):
    def __init__(self, tune=False):
        super(ChineseBERT, self).__init__()
        self.bert = AutoModel.from_pretrained(CHINESE_BERT_NAME)
        if not tune:
            for para in self.bert.parameters():
                para.requires_grad_(False)

    def forward(self, input_ids, token_type_ids, attention_mask):
        return self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=False,
            output_hidden_states=False,
        ).last_hidden_state


class EnglishBERT(nn.Module):
    def __init__(self, tune=False):
        super(EnglishBERT, self).__init__()
        self.bert = AutoModel.from_pretrained(ENGLISH_BERT_NAME)
        if not tune:
            for para in self.bert.parameters():
                para.requires_grad_(False)

    def forward(self, input_ids, token_type_ids, attention_mask):
        return self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=False,
            output_hidden_states=False,
        ).last_hidden_state
