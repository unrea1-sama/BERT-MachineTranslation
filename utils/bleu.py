import math
import collections
import torch


def clean(seqs, tokenizer):
    results = []
    for seq in seqs:
        result = []
        for x in tokenizer.convert_ids_to_tokens(seq):
            if x == "[SEP]":
                # this is a EOS symbol
                break
            else:
                if x != "[PAD]" and x != "[CLS]":
                    # merge word piece
                    if x.startswith('##'):
                        if len(result):
                            result[-1]+=x[2:]
                    else:
                        result.append(x)
        results.append(result)
    return results


def calbleu(pred, ground_truth, k=4):
    """
    calculate average bleu score for one batch
    """
    result = []
    for p, g in zip(pred, ground_truth):
        result.append(bleu(p, g, k))
    return torch.tensor(result).mean()


def bleu(pred_tokens, label_tokens, k):
    """Compute the BLEU."""
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[" ".join(label_tokens[i : i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[" ".join(pred_tokens[i : i + n])] > 0:
                num_matches += 1
                label_subs[" ".join(pred_tokens[i : i + n])] -= 1
        score *= math.pow(num_matches / max(len_pred - n + 1, 1), math.pow(0.5, n))
    return score
