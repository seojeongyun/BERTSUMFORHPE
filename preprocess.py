#encoding=utf-8


import argparse
import time

from others.logging import init_logger
from prepro import data_builder


def do_format_to_bert(args):
    print(time.clock())
    data_builder.format_to_bert(args)
    print(time.clock())

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", default='format_to_bert', type=str)
    parser.add_argument("-format_mode", default='raw', help='[raw/prepro]')
    parser.add_argument("-corpus_type", default="Training")
    parser.add_argument("-data_type", default='law')
    parser.add_argument("-raw_path", default='/storage/hjchoi/Document_Summary_text/')
    parser.add_argument("-json_path", default='/storage/hjchoi/Document_Summary_text/')
    #parser.add_argument("-predata_path", default="/storage/hjchoi/Document_Summary_text/Training/news.tsv")
    parser.add_argument("-save_path", default='./bert_data/law/')
    parser.add_argument("-use_anno_labels", type=str2bool, default=True)

    parser.add_argument("-shard_size", default=2000, type=int)
    parser.add_argument('-min_src_nsents', default=3, type=int)
    parser.add_argument('-max_src_nsents', default=100, type=int)
    parser.add_argument('-min_src_ntokens_per_sent', default=5, type=int) # default=5
    parser.add_argument('-max_src_ntokens_per_sent', default=200, type=int)
    parser.add_argument('-min_tgt_ntokens', default=5, type=int)
    parser.add_argument('-max_tgt_ntokens', default=500, type=int)

    parser.add_argument("-lower", type=str2bool, nargs='?',const=True,default=True)

    parser.add_argument('-log_file', default=None) #'./logs/bertnews.log'
    parser.add_argument('-dataset', default='')
    parser.add_argument('-n_cpus', default=1, type=int)

    args = parser.parse_args()
    if args.log_file is not None:
        init_logger(args.log_file)
    eval('data_builder.'+ args.mode + '(args)')
