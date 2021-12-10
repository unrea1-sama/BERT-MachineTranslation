import torch
from .PositionalEncoding import PositionalEncoding
from torch import nn
from .bert import ChineseBERT, EnglishBERT, BERT_DIM


class MachineTranslationModel(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        transformer_dropout,
        tgt_vocab_size,
        tgt_max_length,
        finetune_src_bert,
        finetune_tgt_bert,
        pe_dropout,
        layer_norm_eps=1e-5,
    ):
        super(MachineTranslationModel, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.transformer_dropout = transformer_dropout
        self.tgt_vocab_size = tgt_vocab_size
        self.tgt_max_length = tgt_max_length

        self.encoder_pe = PositionalEncoding(d_model, pe_dropout)
        self.decoder_pe = PositionalEncoding(d_model, pe_dropout)
        self.src_bert = ChineseBERT(finetune_src_bert)
        self.tgt_bert = EnglishBERT(finetune_tgt_bert)
        self.transformer = nn.Transformer(
            self.d_model,
            self.nhead,
            self.num_encoder_layers,
            self.num_decoder_layers,
            self.dim_feedforward,
            self.transformer_dropout,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
        )
        self.encoder_input_linear = nn.Linear(BERT_DIM, d_model)
        self.decoder_input_linear = nn.Linear(BERT_DIM, d_model)
        self.decoder_output_linear = nn.Linear(d_model, tgt_vocab_size)
        self.softmax = nn.Softmax(dim=2)

        self.decoder_tgt_mask = nn.parameter.Parameter(
            nn.Transformer.generate_square_subsequent_mask(tgt_max_length), False
        )

    def forward(
        self,
        src_input_ids,
        src_token_type_ids,
        src_attention_mask,
        tgt_input_ids,
        tgt_token_type_ids,
        tgt_attention_mask,
    ):
        # src_input_ids: (batch_size,max_src_length)
        # src_token_type_ids: (batch_size,max_src_length)
        # src_attention_mask: (batch_size,max_src_length),
        #   indicates the non-padding parts of the src_input_ids,
        #   where 1 indicates a non-padding elements.
        # tgt_input_ids: (batch_size,max_tgt_length)
        # tgt_token_type_ids: (batch_size,max_tgt_length)
        # tgt_attention_mask: (batch_size,max_tgt_length)
        #   similar to src_attention_mask.

        src_bert_output = self.src_bert(
            src_input_ids, src_token_type_ids, src_attention_mask
        )
        tgt_bert_output = self.tgt_bert(
            tgt_input_ids, tgt_token_type_ids, tgt_attention_mask
        )
        # src_bert_output: (batch_size,max_src_length,BERT_DIM)
        # tgt_bert_output: (batch_size,max_tgt_length,BERT_DIM)

        # the src_key_padding_mask, tgt_key_padding_mask and memory_key_padding_mask
        # uses True to indicate a padding element
        src_bert_output = self.encoder_pe(self.encoder_input_linear(src_bert_output))
        tgt_bert_output = self.decoder_pe(self.decoder_input_linear(tgt_bert_output))
        src_key_padding_mask = src_attention_mask == 0
        tgt_key_padding_mask = tgt_attention_mask == 0
        transformer_output = self.transformer(
            src_bert_output,
            tgt_bert_output,
            tgt_mask=self.decoder_tgt_mask.to(src_bert_output),
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        return self.softmax(self.decoder_output_linear(transformer_output))
