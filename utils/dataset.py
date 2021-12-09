from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoTokenizer

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
        temp_list = []
        # if the target corpus is given, generate training examples
        if tgt_filepath:
            self.collate_fn = TranslationTrainingDatasetCollateFn(
                src_tokenizer_name,
                tgt_tokenizer_name,
                src_max_length,
                tgt_max_length,
            )
            with open(src_filepath) as src_file:
                with open(tgt_filepath) as tgt_file:
                    for src_line, tgt_line in zip(src_file, tgt_file):
                        # manually add BOS and EOS token into the target sentence
                        # since the input to decoder requires BOS
                        # and the prediction of the decoder requires EOS
                        temp_list.append(
                            [
                                src_line,
                                " ".join([BOS, tgt_line]),
                                " ".join([tgt_line, EOS]),
                            ]
                        )

        # no target corpus, generate test examples for inference
        else:
            self.collate_fn = TranslationInferenceDatasetCollateFn(
                src_tokenizer_name, tgt_tokenizer_name, src_max_length, tgt_max_length
            )
            with open(src_filepath) as src_file:
                for src_line in src_file:
                    # BOS symbol here is just a start symbol for inference
                    temp_list.append([src_line, BOS])
        self.dataset = np.array(temp_list)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return self.dataset.shape[0]


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
        )
        # we have already add special tokens for tgt_input and tgt_output
        tgt_input_token = self.tgt_tokenizer(
            batch_tgt_input,
            return_tensors="pt",
            max_length=self.tgt_max_length,
            padding="max_length",
            add_special_tokens=False,
        )
        tgt_output_token = self.tgt_tokenizer(
            batch_tgt_output,
            return_tensors="pt",
            max_length=self.tgt_max_length,
            padding="max_length",
            add_special_tokens=False,
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
        )
        # we have already add special tokens for tgt_input and tgt_output
        tgt_input_token = self.tgt_tokenizer(
            batch_tgt_input,
            return_tensors="pt",
            max_length=self.tgt_max_length,
            padding="max_length",
            add_special_tokens=False,
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
    num_workers,
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
        num_workers=num_workers,
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
