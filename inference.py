import torch

from model import MachineTranslationModel, CHINESE_BERT_NAME, ENGLISH_BERT_NAME
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--src_file", type=str)
    parser.add_argument("--tgt_file", type=str)
    parser.add_argument("--src_max_length", type=int)
    parser.add_argument("--tgt_max_length", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_dataloader_worker", type=int)
    parser.add_argument("--d_model", type=int)
    parser.add_argument("--nhead", type=int)
    parser.add_argument("--num_encoder_layers", type=int)
    parser.add_argument("--num_decoder_layers", type=int)
    parser.add_argument("--dim_feedforward", type=int)
    parser.add_argument("--transformer_dropout", type=float)
    parser.add_argument("--pe_dropout",type=float)
    parser.add_argument("--tgt_vocab_size", type=int)
    return parser.parse_args()



def main(args):
    pass


if __name__ =='__main__':
    main(parse_args())