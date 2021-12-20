import torch
from torch import nn, optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.args import parse_args
from utils.dataset import load_dataset
from utils.loss import MaskedMSELoss
from model import MachineTranslationModel, CHINESE_BERT_NAME, ENGLISH_BERT_NAME
import os
from utils.lr import set_up_optimizer
from torch.utils.tensorboard import SummaryWriter
from utils.bleu import calbleu, clean


def main(rank, args):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    dist.init_process_group("nccl", rank=rank, world_size=args.device_count)

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
    ).to(rank)
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    loss_fn = MaskedMSELoss()
    train_dataloader = load_dataset(
        args.train_src_file,
        args.train_tgt_file,
        CHINESE_BERT_NAME,
        ENGLISH_BERT_NAME,
        args.batch_size,
        args.src_max_length,
        args.tgt_max_length,
        args.shuffle_training_set,
        args.device_count,
        rank,
    )
    dev_dataloader = load_dataset(
        args.dev_src_file,
        args.dev_tgt_file,
        CHINESE_BERT_NAME,
        ENGLISH_BERT_NAME,
        args.batch_size,
        args.src_max_length,
        args.tgt_max_length,
        False,
        1,
        None,
    )  # dev_dataloader are not required to use distributed sampler
    optimizer, lr_scheduler = set_up_optimizer(
        ddp_model, optim.Adam, args.d_model, args.warmup_step
    )
    if rank == 0:
        writer = SummaryWriter(args.log_dir)
    dist.barrier()
    step = 1
    eval_step = 1
    for i in range(args.epoch):
        if args.device_count > 1:
            train_dataloader.sampler.set_epoch(i)
        ddp_model.train()
        for (
            src_input_ids,
            src_token_type_ids,
            src_attention_mask,
            tgt_input_ids,
            tgt_token_type_ids,
            tgt_attention_mask,
            tgt_output_ids,
            tgt_output_mask,
            _,
            _,
            _,
        ) in train_dataloader:
            optimizer.zero_grad()
            pred = ddp_model(
                src_input_ids,
                src_token_type_ids,
                src_attention_mask,
                tgt_input_ids,
                tgt_token_type_ids,
                tgt_attention_mask,
            )
            tgt_output_ids = tgt_output_ids.to(pred.device)
            tgt_output_mask = tgt_output_mask.to(pred.device)
            loss = loss_fn(pred.permute(0, 2, 1), tgt_output_ids, tgt_output_mask)
            loss.backward()
            optimizer.step()
            if rank == 0:
                print(
                    "training: epoch {}, total step: {}, lr: {}, loss: {}".format(
                        i + 1, step, lr_scheduler.get_last_lr()[0], loss.item()
                    )
                )
                writer.add_scalar("train loss", loss.item(), step)
                writer.add_scalar("learning rate", lr_scheduler.get_last_lr()[0], step)

            if step % args.eval_step == 0:
                ddp_model.eval()
                with torch.no_grad():
                    all_decoded_result = []
                    all_decoded_target = []
                    for (
                        src_input_ids,
                        src_token_type_ids,
                        src_attention_mask,
                        tgt_input_ids,
                        tgt_token_type_ids,
                        tgt_attention_mask,
                        tgt_output_ids,
                        tgt_output_mask,
                        _,
                        _,
                        _,
                    ) in dev_dataloader:
                        pred = ddp_model(
                            src_input_ids,
                            src_token_type_ids,
                            src_attention_mask,
                            tgt_input_ids,
                            tgt_token_type_ids,
                            tgt_attention_mask,
                        )
                        tgt_output_ids = tgt_output_ids.to(pred.device)
                        tgt_output_mask = tgt_output_mask.to(pred.device)
                        loss = loss_fn(
                            pred.permute(0, 2, 1), tgt_output_ids, tgt_output_mask
                        )
                        decoded_result = dev_dataloader.dataset.collate_fn.tgt_tokenizer.batch_decode(
                            pred.argmax(dim=2), skip_special_tokens=False
                        )
                        decoded_ground_truth = dev_dataloader.dataset.collate_fn.tgt_tokenizer.batch_decode(
                            tgt_output_ids, skip_special_tokens=False
                        )
                        all_decoded_result.extend(decoded_result)
                        all_decoded_target.extend(decoded_ground_truth)
                        bleu = calbleu(decoded_result, decoded_ground_truth)
                        if rank == 0:
                            writer.add_scalar("dev loss", loss.item(), eval_step)
                            writer.add_scalar("dev bleu", bleu, eval_step)
                            print(
                                "eval: epoch {}, eval step: {}, loss: {}, bleu: {}".format(
                                    i + 1, eval_step, loss.item(), bleu
                                )
                            )

                        eval_step += 1
                if rank == 0:
                    torch.save(
                        model.state_dict(),
                        os.path.join(
                            args.log_dir, "checkpoint_{}_{}.pt".format(i + 1, eval_step)
                        ),
                    )
                    with open(
                        os.path.join(
                            args.log_dir, "result_{}_{}.txt".format(i + 1, eval_step)
                        ),
                        "w",
                    ) as f:
                        f.writelines(
                            [" ".join(clean(x)) + "\n" for x in all_decoded_result]
                        )
                    with open(
                        os.path.join(args.log_dir, "ground_truth_{}_{}.txt").format(
                            i + 1, eval_step
                        ),
                        "w",
                    ) as f:
                        f.writelines(
                            [" ".join(clean(x)) + "\n" for x in all_decoded_target]
                        )
                    for name, param in ddp_model.named_parameters():
                        if "transformer" in name:
                            writer.add_histogram(name, param, step)
            # to make all process synchronized
            dist.barrier()
            lr_scheduler.step()
            step += 1

    dist.destroy_process_group()
    if rank == 0:
        writer.close()


if __name__ == "__main__":
    args = parse_args()
    mp.spawn(main, args=(args,), nprocs=args.device_count, join=True)
