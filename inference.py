import torch
from utils.dataset import load_dataset
from model import MachineTranslationModel, CHINESE_BERT_NAME, ENGLISH_BERT_NAME
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--src_file", type=str)
    parser.add_argument("--tgt_file", type=str, help="file for translation result")
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
    parser.add_argument("--pe_dropout", type=float)
    parser.add_argument("--tgt_vocab_size", type=int)
    parser.add_argument("--device", type=int)
    parser.add_argument("--beam_width", type=1)
    return parser.parse_args()


def main(args):
    model = MachineTranslationModel(
        args.d_model,
        args.nhead,
        args.num_encoder_layers,
        args.num_decoder_layers,
        args.dim_feedforward,
        args.transformer_dropout,
        args.tgt_vocab_size,
        args.tgt_max_length,
        args.finetune_src_bert,
        args.finetune_tgt_bert,
        args.pe_dropout,
    ).to(args.device)
    eval_dataloader = load_dataset(
        args.src_file,
        None,
        CHINESE_BERT_NAME,
        ENGLISH_BERT_NAME,
        args.batch_size,
        args.src_max_length,
        args.tgt_max_length,
        False,
        1,
    )
    decoded_outputs = []
    for (
        src_input_ids,
        src_token_type_ids,
        src_attention_mask,
        tgt_input_ids,
        tgt_token_type_ids,
        tgt_attention_mask,
    ) in eval_dataloader:
        # predict autoregressively
        # src_input_ids: (batch_size,src_max_length)
        src_input_ids = src_input_ids.to(args.device)
        src_token_type_ids = src_token_type_ids.to(args.device)
        src_attention_mask = src_attention_mask.to(args.device)
        tgt_input_ids = tgt_input_ids.to(args.device)
        tgt_token_type_ids = tgt_token_type_ids.to(args.device)
        tgt_attention_mask = tgt_attention_mask.to(args.device)

        preds = model(
            src_input_ids,
            src_token_type_ids,
            src_attention_mask,
            tgt_input_ids,
            tgt_token_type_ids,
            tgt_attention_mask,
        )
        # preds: (batch_size,tgt_max_length,tgt_vocab_size)

        beam_prob, topk_ids = preds.topk(args.beam_width, dim=2)
        # topk_ids: (batch_size,tgt_max_length,beam_width)
        # beam_prob: (batch_size,tgt_max_length,beam_width)

        def generate_beam_data(t):
            # t: (batch_size,length)
            # return shape: (batch_size*beam_width,length)
            return torch.tile(t, (1, args.beam_width)).reshape((-1, t.shape[1]))

        def extend_last_dim_by_one(t):
            x, y = t.shape
            temp = torch.zeros((x, y + 1), device=t.device)
            temp[:, :y] = t
            return t

        beam_src_input_ids = generate_beam_data(src_input_ids)
        beam_src_token_type_ids = generate_beam_data(src_token_type_ids)
        beam_src_attention_mask = generate_beam_data(src_attention_mask)
        beam_tgt_input_ids = generate_beam_data(tgt_input_ids)
        beam_tgt_token_type_ids = generate_beam_data(tgt_token_type_ids)
        beam_tgt_attention_mask = generate_beam_data(tgt_attention_mask)
        # beam_src_input_ids, beam_src_token_type_ids, beam_src_attention_mask:
        # (batch_size*beam_width,src_max_length)
        # beam_tgt_input_ids, beam_tgt_token_type_ids, beam_tgt_attention_mask:
        # (batch_size*beam_width,tgt_max_length)
        beam_tgt_input_ids[:, 1] = topk_ids[:, 0].flatten()
        beam_tgt_attention_mask[:, 1] = 1

        total_beam_prob = beam_prob[:, 0].flatten().reshape((-1, 1))
        # total_beam_prob: (batch_size*beam_width,1)

        beam_tgt_input_ids = extend_last_dim_by_one(beam_tgt_input_ids)
        beam_tgt_token_type_ids = extend_last_dim_by_one(beam_tgt_token_type_ids)
        beam_tgt_attention_mask = extend_last_dim_by_one(beam_tgt_attention_mask)

        for i in range(1, args.tgt_max_length):
            preds = model(
                beam_src_input_ids,
                beam_src_token_type_ids,
                beam_src_attention_mask,
                beam_tgt_input_ids,
                beam_tgt_token_type_ids,
                beam_tgt_attention_mask,
            )
            # preds: (batch_size*beam_width,tgt_max_length,tgt_vocab_size)

            preds = preds[:, i]
            # preds: (batch_size*beam_width,tgt_vocab_size), probability to next node of current step
            temp_beam_prob = total_beam_prob * preds
            # temp_beam_prob: (batch_size*beam_width,tgt_vocab_size)
            temp_beam_prob = temp_beam_prob.reshape((args.batch_size, -1))
            # temp_beam_prob: (batch_size,beam_width*tgt_vocab_size)
            p, idx = temp_beam_prob.topk(args.beam_width, dim=1)
            # p, idx: (batch_size,beam_width)
            temp_idx = idx % args.tgt_vocab_size

            total_beam_prob = p.reshape((-1, 1))
            beam_tgt_attention_mask[:, i + 1] = 1

            # to get the current best path
            temp_path = (
                (idx / args.tgt_vocab_size)
                .unsqueeze(2)
                .tile((1, 1, args.tgt_max_length))
            )
            # temp_path: (batch_size,beam_width,tgt_max_length)
            best_beam_tgt_input_ids = beam_tgt_input_ids.gather(dim=1, index=temp_path)
            best_beam_tgt_input_ids = best_beam_tgt_input_ids.reshape(
                (-1, args.tgt_max_length)
            )
            # set the best decoding result of the current step
            best_beam_tgt_input_ids[:, i + 1] = temp_idx.flatten()
            # update beam_tgt_input_ids
            beam_tgt_input_ids = best_beam_tgt_input_ids
        decoded_outputs.extend(
            eval_dataloader.collate_fn.tgt_tokenizer.batch_decode(beam_tgt_input_ids)
        )
    with open(args.tgt_file, "w") as f:
        f.writelines([x + "\n" for x in decoded_outputs])


if __name__ == "__main__":
    main(parse_args())
