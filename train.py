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
        args.num_dataloader_worker,
    )
    dev_dataloader = load_dataset(
        args.dev_src_file,
        args.dev_tgt_file,
        CHINESE_BERT_NAME,
        ENGLISH_BERT_NAME,
        args.batch_size,
        args.src_max_length,
        args.tgt_max_length,
        True,
        args.num_dataloader_worker,
    )
    optimizer, lr_scheduler = set_up_optimizer(
        ddp_model, optim.Adam, args.d_model, args.warmup_step
    )
    if rank == 0:
        writer = SummaryWriter(args.log_dir)
    dist.barrier()
    step = 1
    for i in range(args.epoch):
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
            loss = loss_fn(pred.premute(0, 2, 1), tgt_output_ids, tgt_output_mask)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            if rank == 0:
                print(
                    "training: epoch {}, total step: {}, loss: {}".format(
                        i + 1, step, loss.item()
                    )
                )
                writer.add_scalar("train_loss", loss.item(), step)
            step += 1
            # to make all process synchronized
            dist.barrier()
        ddp_model.eval()
        if rank == 0:
            with torch.no_grad():
                for (
                    src_input_ids,
                    src_token_type_ids,
                    src_attention_mask,
                    tgt_input_ids,
                    tgt_token_type_ids,
                    tgt_attention_mask,
                    tgt_output_ids,
                    tgt_output_mask,
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
                    writer.add_scalar("dev_loss", loss.item(), i + 1)
                    print(
                        "eval: epoch {}, total step: {}, loss: {}".format(
                            i + 1, step, loss.item()
                        )
                    )
        torch.save(
            model.state_dict(),
            os.path.join(args.log_dir, "checkpoint-{}.pt".format(i + 1)),
        )
        dist.barrier()

    dist.destroy_process_group()
    writer.close()


if __name__ == "__main__":
    args = parse_args()
    mp.spawn(main, args=(args,), nprocs=args.device_count, join=True)
