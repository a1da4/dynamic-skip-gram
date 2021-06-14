import argparse
import logging
from tqdm import tqdm
import numpy as np

from ioutils import *

def obtain_vocab(files, threshold=100, size=10000):
    """obtain target vocaburaly
    :param files:
    :return: target_vocab
    """
    vocab = []
    id2freq = []
    for file in files:
        logging.info(f" [obtain_vocab] counting {file} ...")
        with open(file) as fp:
            for line in fp:
                words, year, freq, _ = line.strip().split("\t")
                words = words.split()
                if 1800 <= int(year) <= 2008:
                    for word in words:
                        if word not in vocab:
                            vocab.append(word)
                            id2freq.append(0)
                        id = vocab.index(word)
                        id2freq[id] += int(freq)
        logging.info(" [obtain_vocab] finished!")
        logging.debug(f" [obtain_vocab] vocab: {vocab[:10]} ...")
        logging.debug(f" [obtain_vocab] id2freq: {id2freq[:10]} ...")
    logging.debug(f" [obtain_vocab] vocab: {len(vocab)} words: \n{vocab[:10]} ...")

    target_ids = [freq >= threshold for freq in id2freq]
    logging.debug(f" [obtain_vocab] target_ids: {target_ids[:10]} ...")

    target_vocab = [vocab[id] for id in range(len(vocab)) if target_ids[id]]
    logging.debug(
        f" [obtain_vocab] target_vocab: {len(target_vocab)} words: \n{target_vocab[:10]} ..."
    )
    return target_vocab


#def sample_positive(files, vocab):
def sample_positive(files, vocab, time_span):
    """
    :param files:
    :param vocab:
    :param time_span:
    :return: positive_samples (T, V, V)
    """
    positive_samples = []
    set_vocab = set(vocab)
    #for t in range(1800, 2008 + 1):
    for t in range(time_span):
        ps_tmp = [[0 for _ in range(len(vocab))] for _ in range(len(vocab))]
        positive_samples.append(ps_tmp)
        del ps_tmp
    logging.debug(
        f" [sample_positive] positive_samples: ({len(positive_samples)}, {len(positive_samples[0])}, {len(positive_samples[0][0])})"
    )

    for file in files:
        logging.info(f" [sample_positive] counting {file} ...")
        with open(file) as fp:
            for line in fp:
                words, year, freq, _ = line.strip().split("\t")
                words = words.split()
                if 1800 <= int(year) <= 2008:
                    t = int(year) - 1800
                    for i in range(len(words)):
                        target = words[i]
                        if target not in set_vocab:
                            continue
                        target_id = vocab.index(target)
                        for j in range(len(words)):
                            context = words[j]
                            if j == i or context not in set_vocab:
                                continue
                            context_id = vocab.index(context)
                            positive_samples[t][target_id][context_id] += int(freq)
        logging.info(" [sample_positive] finished!")
        logging.debug(
            f" [sample_positive] positive_samples: {positive_samples[0][0][:10]} ..."
        )
    return positive_samples

#TODO mutliprocess with scipy.sparce?
def p_t(i, total_freq_t, positive_samples_t):
    return sum(positive_samples_t[i]) / total_freq_t

def p_t_smoothed(j, total_freq_t, positive_samples_t):
    pj_t_smoothed = p_t(j, total_freq_t, positive_samples_t)**0.75
    p_t_dist_smoothed = 0
    for i in range(len(positive_samples_t)):
        p_t_dist_smoothed += p_t(i, total_freq_t, positive_samples_t)**0.75
    return pj_t_smoothed / p_t_dist_smoothed

def sample_negative(vocab, positive_samples, num_samples, time_span):
    """negative sampling
    :param files:
    :param vocab:
    :param positive_samples: co-occur matrix (T,V,V)
    :param num_samples:
    :param time_span:
    :return: negative_samples, (T, V, V)
    """
    negative_samples = []
    #for t in range(1800, 2008 + 1):
    for t in range(time_span):
        ns_tmp = [[0 for _ in range(len(vocab))] for _ in range(len(vocab))]
        negative_samples.append(ns_tmp)
        del ns_tmp
    logging.debug(
        f" [sample_negative] negative_samples: ({len(negative_samples)}, {len(negative_samples[0])}, {len(negative_samples[0][0])})"
    )

    #for t in range(1800, 2008 + 1):
    for t in tqdm(range(time_span)):
        total_freq_t = sum([sum(ps_tmp) for ps_tmp in positive_samples[t]])
        for i in tqdm(range(len(vocab))):
            pi_t = p_t(i, total_freq_t, positive_samples[t]) 
            for j in tqdm(range(len(vocab))):
                pj_t_smoothed = p_t_smoothed(j, total_freq_t, positive_samples[t])
                ns_tmp = total_freq_t*num_samples*pi_t*pj_t_smoothed
    logging.info(" [sample_negative] finished!")
    logging.debug(
        f" [sample_negative] negative_samples: {negative_samples[0][0][:10]} ..."
    )


def preprocess(args):
    # logging.basicConfig(filename="preprocess.log", filemode="w", level=logging.DEBUG)
    logging.basicConfig(level=logging.DEBUG)
    logging.debug(f" args: \n{args}")
    if args.vocab is None:
        vocab = obtain_vocab(args.files)
        save_vocab(vocab)
    else:
        vocab = load_vocab(args.vocab)
    # positive/negative samples を獲得する
    # positive は愚直にカウントし、 negative は appendix 2. より算出
    if args.positive is None:
        #positive_samples = sample_positive(args.files, vocab)
        positive_samples = sample_positive(args.files, vocab, args.time_span)
        save_3d_matrix(positive_samples, name="positive_samples")
    else:
        #positive_samples = load_3d_matrix(args.positive, z_size=209, x_size=len(vocab), y_size=len(vocab))
        positive_samples = load_3d_matrix(args.positive, z_size=args.time_span, x_size=len(vocab), y_size=len(vocab))
    #negative_samples = sample_negative(vocab, positive_samples)
    negative_samples = sample_negative(vocab, positive_samples, args.num_samples, args.time_span)


def cli_preprocess():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="*", help="path of files")
    parser.add_argument("--vocab", help="path of vocab")
    parser.add_argument("--positive", help="path of positive samples")
    parser.add_argument("--time_span", type=int, help="time span (e.g. 1800-2008: 209)")
    parser.add_argument("--num_samples", type=int, default=1, help="num of negative samples")
    args = parser.parse_args()
    preprocess(args)


if __name__ == "__main__":
    cli_preprocess()
