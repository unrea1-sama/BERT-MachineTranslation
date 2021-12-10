from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm
import h5py

BOS = "[CLS]"
EOS = "[SEP]"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--src_file", type=str)
    parser.add_argument("--tgt_file", type=str)
    parser.add_argument("--src_output_file", type=str)
    parser.add_argument("--tgt_output_file", type=str)
    return parser.parse_args()


def create_h5_dataset(input_file, output_file):
    ofile = h5py.File(output_file, "w")
    with open(input_file) as ifile:
        ofile["data"] = ifile.readlines()
    ofile.close()


def main(args):
    if args.src_file:
        create_h5_dataset(args.src_file, args.src_output_file)
    if args.tgt_file:
        create_h5_dataset(args.tgt_file, args.tgt_output_file)
    print("finish preprocessing files!")


if __name__ == "__main__":
    main(parse_args())
