import re
from collections import defaultdict
import csv
from argparse import ArgumentParser

# pipeline：
# 断句->替换->移除拟声词->->分词

hotword = defaultdict(str)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_path", type=str, help="path to input file to be translated, should not be segmented"
    )
    parser.add_argument("--output_path", type=str, help="path to output file")
    parser.add_argument("--hotword_path", type=str, default="hot_word.csv")
    return parser.parse_args()


def load_hotword(path):
    """
    读取热词
    """
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row[1].strip()):
                hotword[row[0].strip()] = row[1].strip()


def replace(line, key, value):
    line = re.sub(" " + key, " " + value, line)
    line = re.sub(key + " ", value + " ", line)
    return line


def replace_hotword(line):
    """
    替换热词
    """
    for key in hotword:
        line = replace(line, key, hotword[key])
    return line


def remove_pause_word(line):
    """
    删除拟声词
    """
    return re.sub("[嗯啊呀嘛吗吧哦哟呃]", "", line)


def split_word(line):
    """
    分词，使用jieba
    """
    import jieba
    return ' '.join(list(jieba.cut(line)))


def split_line(line):
    """
    断句，还没做...
    """
    return line


pipeline = [split_line, replace_hotword, remove_pause_word, split_word]


def main(args):
    load_hotword(args.hotword_path)
    with open(args.input_path) as input_file:
        with open(args.output_path, "w") as output_file:
            for line in input_file:
                for funciton in pipeline:
                    line = funciton(line)
                output_file.write(line)


if __name__ == "__main__":
    main(parse_args())
