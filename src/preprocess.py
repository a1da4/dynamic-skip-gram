import argparse
import logging
import os

import numpy as np
from tqdm import tqdm

from ioutils import *


def obtain_vocab(files, vocab_size=10000):
    """obtain target vocaburaly from document in each time periods
    :param files: list, path of files
    :param vocab_size: int, vocab size
    :return: target_vocab
    """
    vocab = []
    id2freq = []
    for file in tqdm(files, "obtain vocab..."):
        logging.info(f" [obtain_vocab] # counting {file} ...")
        with open(file) as fp:
            for line in fp:
                words = line.strip().split()
                for word in words:
                    if word not in vocab:
                        vocab.append(word)
                        id2freq.append(0)
                    id = vocab.index(word)
                    id2freq[id] += 1
        logging.info(" [obtain_vocab] ## finished!")
        logging.debug(f" [obtain_vocab] ## vocab (size): {len(vocab)}")
        logging.debug(f" [obtain_vocab] ## total freq: {sum(id2freq)}")
    logging.debug(f" [obtain_vocab] # vocab: {len(vocab)} words: \n{vocab[:10]} ...")

    target_ids = set(np.argsort(-1 * np.array(id2freq))[:size])

    target_vocab = [vocab[id] for id in range(len(vocab)) if id in target_ids]
    logging.debug(f" [obtain_vocab] # target_vocab: {len(target_vocab)} words")
    return target_vocab


def sample_positive(files, vocab, window_size=4):
    """
    :param files:
    :param vocab:
    :param window_size:
    :return: positive_samples (T, V, V)
    """
    positive_samples = []
    set_vocab = set(vocab)
    logging.info(" [sample_positive] # initialize positive samples ...")
    for t in range(len(files)):
        ps_tmp = [[0 for _ in range(len(vocab))] for _ in range(len(vocab))]
        positive_samples.append(ps_tmp)
        del ps_tmp
    logging.info(f" [sample_positive] # finished! (size): ({len(positive_samples)}, {len(positive_samples[0])}, {len(positive_samples[0][0])}")

    for t, file in tqdm(enumerate(files), "obtrain positive samples..."):
        logging.info(f" [sample_positive] # counting {file} ...")
        with open(file) as fp:
            for line in fp:
                words = line.strip().split()
                for target_id in range(len(words)):
                    target_word = words[target_id]
                    if target_word not in set_vocab:
                        continue
                    target_id = vocab.index(target_word)
                    
                    for shift in range(1, window_size + 1):
                        left_id = target_id - shift
                        right_id = target_id + shift

                        if left_id >= 0:
                            left_context_word = words[left_id]
                            left_context_id = vocab.index(left_context_word)
                            positive_samples[t][target_id][left_context_id] += 1
                        
                        if right_id < len(words):
                            right_context_word = words[right_id]
                            right_context_id = vocab.index(right_context_word)
                            positive_samples[t][target_id][right_context_id] += 1
                        
            logging.info(" [sample_positive] ## finished!")
    return positive_samples


def sample_negative(vocab, positive_samples, num_samples):
    """negative sampling
    :param vocab:
    :param positive_samples: co-occur matrix (T,V,V)
    :param num_samples: int, number of negative sample(s)
    """
    os.makedirs(f"../negative_samples", exist_ok=True)

    for t in tqdm(len(positive_samples), "obtain negative samples..."):
        # total_freq_t: single value
        total_freq_t = sum([sum(ps_tmp) for ps_tmp in positive_samples[t]])
        # freq_matrix_t: (V, V) matrix
        freq_matrix_t = np.array(positive_samples[t])
        # Pt: (V), maximum likelihood of target word occurs 
        Pt = np.sum(freq_matrix_t, axis=1) / total_freq_t
        # Pt_smoothed: (V), smoothed probablity
        Pt_smoothed = Pt**0.75
        Pt_dash = Pt_smoothed / np.sum(Pt_smoothed)
        del Pt_smoothed
        # negative_samples_t: (V, V)
        negative_samples_t = total_freq_t * Pt.reshape(-1, 1) * Pt_dash.reshape(1, -1)
        save_2d_matrix(negative_samples_t, name=f"negative_samples/{t}")

    logging.info(" [sample_negative] # finished!")


def preprocess(args):
    #logging.basicConfig(filename="preprocess.log", filemode="w", level=logging.DEBUG)
    logging.basicConfig(filename="../preprocess.log", filemode="w", level=logging.DEBUG)
    logging.debug(f" args: \n{args}")
    
    logging.info(" [preprocess] Obtain files ...") 
    if args.filedir[-1] != "/":
        args.filedir += "/"
    files = [args.filedir + file for file in os.listdir(args.filedir)]
    logging.debug(f" [preprocess] # {len(files)} files")
    logging.debug(f" [preprocess] # example {files[0]}")

    logging.info(" [preprocess] Obtain vocab ...")
    if args.vocab is None:
        vocab = obtain_vocab(files, vocab_size=args.size)
        save_vocab(vocab)
    else:
        vocab = load_vocab(args.vocab)
    
    logging.info(" [preprocess] Obtain positive samples ...")

    if args.positive is None:
        positive_samples = sample_positive(files, vocab, args.window)
        save_3d_matrix(positive_samples, name="positive_samples")
    else:
        positive_samples = load_3d_matrix(
            args.positive, z_size=len(files), x_size=len(vocab), y_size=len(vocab)
        )

    logging.info(" [preprocess] Obtain negative samples ...")
    sample_negative(
        vocab, positive_samples, args.num_samples
    )


def cli_preprocess():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filedir", help="path of file dir")
    parser.add_argument("--size", type=int, default=10000, help="size of vocab")
    parser.add_argument("--vocab", help="path of vocab")
    parser.add_argument("--window", type=int, default=4, help="window size")
    parser.add_argument("--positive", help="path of positive samples")
    parser.add_argument(
        "--num_samples", type=int, default=1, help="num of negative samples"
    )
    args = parser.parse_args()
    preprocess(args)


if __name__ == "__main__":
    cli_preprocess()
