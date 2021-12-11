from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoTokenizer
import h5py

BOS = "[CLS]"
EOS = "[SEP]"


class TranslationDataset(Dataset):
    def __init__(
        self,
        src_tokenizer_name,
        tgt_tokenizer_name,
        src_max_length,
        tgt_max_length,
        src_filepath,
        tgt_filepath=None,
    ):
        super(TranslationDataset, self).__init__()
        self.src_h5 = h5py.File(src_filepath, "r")
        # if the target corpus is given, generate training examples
        if tgt_filepath:
            self.collate_fn = TranslationTrainingDatasetCollateFn(
                src_tokenizer_name,
                tgt_tokenizer_name,
                src_max_length,
                tgt_max_length,
            )
            self.tgt_h5 = h5py.File(tgt_filepath, "r")
        # no target corpus, generate test examples for inference
        else:
            self.collate_fn = TranslationInferenceDatasetCollateFn(
                src_tokenizer_name, tgt_tokenizer_name, src_max_length, tgt_max_length
            )
            self.tgt_h5 = None

    def __getitem__(self, index):
        src = self.src_h5["data"][index].decode("utf-8")
        if self.tgt_h5:
            tgt = self.tgt_h5["data"][index].decode("utf-8")
            return [src, " ".join([BOS, tgt]), " ".join([tgt, EOS])]
        else:
            return [src, BOS]

    def __len__(self):
        return self.src_h5["data"].len()


class TranslationTrainingDatasetCollateFn:
    def __init__(
        self, src_tokenizer_name, tgt_tokenizer_name, src_max_length, tgt_max_length
    ):
        self.src_tokenizer = AutoTokenizer.from_pretrained(src_tokenizer_name)
        self.tgt_tokenizer = AutoTokenizer.from_pretrained(tgt_tokenizer_name)
        self.src_max_length = src_max_length
        self.tgt_max_length = tgt_max_length

    def __call__(self, batch_data):
        batch_src_input = [sample[0] for sample in batch_data]
        batch_tgt_input = [sample[1] for sample in batch_data]
        batch_tgt_output = [sample[2] for sample in batch_data]

        src_token = self.src_tokenizer(
            batch_src_input,
            return_tensors="pt",
            max_length=self.src_max_length,
            padding="max_length",
            truncation=True,
        )
        # we have already add special tokens for tgt_input and tgt_output
        tgt_input_token = self.tgt_tokenizer(
            batch_tgt_input,
            return_tensors="pt",
            max_length=self.tgt_max_length,
            padding="max_length",
            add_special_tokens=False,
            truncation=True,
        )
        tgt_output_token = self.tgt_tokenizer(
            batch_tgt_output,
            return_tensors="pt",
            max_length=self.tgt_max_length,
            padding="max_length",
            add_special_tokens=False,
            truncation=True,
        )

        return (
            src_token["input_ids"],
            src_token["token_type_ids"],
            src_token["attention_mask"],
            tgt_input_token["input_ids"],
            tgt_input_token["token_type_ids"],
            tgt_input_token["attention_mask"],
            tgt_output_token["input_ids"],
            tgt_output_token["attention_mask"],
        )


class TranslationInferenceDatasetCollateFn:
    def __init__(
        self, src_tokenizer_name, tgt_tokenizer_name, src_max_length, tgt_max_length
    ):
        self.src_tokenizer = AutoTokenizer.from_pretrained(src_tokenizer_name)
        self.tgt_tokenizer = AutoTokenizer.from_pretrained(tgt_tokenizer_name)
        self.src_max_length = src_max_length
        self.tgt_max_length = tgt_max_length

    def __call__(self, batch_data):
        # dataset for inference doesn't have the ground truth
        batch_src_input = [sample[0] for sample in batch_data]
        batch_tgt_input = [sample[1] for sample in batch_data]

        src_token = self.src_tokenizer(
            batch_src_input,
            return_tensors="pt",
            max_length=self.src_max_length,
            padding="max_length",
            truncation=True,
        )
        # we have already add special tokens for tgt_input and tgt_output
        tgt_input_token = self.tgt_tokenizer(
            batch_tgt_input,
            return_tensors="pt",
            max_length=self.tgt_max_length,
            padding="max_length",
            add_special_tokens=False,
            truncation=True,
        )

        return (
            src_token["input_ids"],
            src_token["token_type_ids"],
            src_token["attention_mask"],
            tgt_input_token["input_ids"],
            tgt_input_token["token_type_ids"],
            tgt_input_token["attention_mask"],
        )


def load_dataset(
    src_filepath,
    tgt_filepath,
    src_bert_name,
    tgt_bert_name,
    batch_size,
    src_max_length,
    tgt_max_length,
    shuffle,
):
    dataset = TranslationDataset(
        src_bert_name,
        tgt_bert_name,
        src_max_length,
        tgt_max_length,
        src_filepath,
        tgt_filepath,
    )
    return DataLoader(
        dataset,
        batch_size,
        shuffle,
        collate_fn=dataset.collate_fn,
    )


if __name__ == "__main__":
    for i in load_dataset(
        "corpus/dev.zh",
        "corpus/dev.en",
        "bert-base-chinese",
        "bert-base-cased",
        128,
        256,
        512,
        True,
        1,
    ):
        print(i)
        break
