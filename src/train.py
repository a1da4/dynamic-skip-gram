import argparse
import logging
import pickle

import numpy as np

from ioutils import *
from model import *


def main(args):
    #logging.basicConfig(filename="../train.log", filemode="w", level=logging.INFO)
    logging.basicConfig(filename="../train_debug.log", filemode="w", level=logging.DEBUG)

    logging.info(" [main] Initialization ...")
    vocab = load_vocab(args.vocab)
    num_timebins = (args.time_end - args.time_start) // args.time_span + 1

    dwe = SkipGramSmoothing(
        seed=args.seed,
        vocab=vocab,
        T=num_timebins,
        dim=args.dim,
        D=args.diffusion,
        taus=range(args.time_start, args.time_end + 1, args.time_span),
    )
    logging.debug(" [main] # model: SkipGramSmoothing")
    logging.debug(f" [main] # dwe.vocab: {len(dwe.vocab)} words")
    logging.debug(f" [main] # dwe.T: {dwe.T}")
    logging.debug(f" [main] # dwe.D: {dwe.D}")
    logging.debug(f" [main] # dwe.mean_target: {dwe.mean_target.shape}")
    logging.debug(f" [main] # dwe.mean_context: {dwe.mean_context.shape}")
    logging.debug(f" [main] # dwe.vals: {dwe.vals}")
    logging.debug(f" [main] # dwe.precision: \n{dwe.precision}")
    logging.debug(f" [main] # dwe.v (size): {dwe.v_target.shape}")
    logging.debug(f" [main] # dwe.v (dtype): {dwe.v_target.dtype}")
    logging.debug(f" [main] # dwe.v (value): \n{dwe.v_target[0][0]}")
    logging.debug(f" [main] # dwe.w (size): {dwe.w_target.shape}")
    logging.debug(f" [main] # dwe.w (dtype): {dwe.w_target.dtype}")
    logging.debug(f" [main] # dwe.w (value): \n{dwe.w_target[0][0]}")

    dataloader = DataLoader(len(vocab), num_timebins, args.positive, args.negative)

    logging.info(" [main] Pre-training ...")
    dwe.train(dataloader, iter=args.pretrain_iter, alpha=args.pretrain_alpha, rate=0.1)
    pickle.dump(dwe, open(f"../dwe_pretrained_ckpt-{args.pretrain_iter}.pkl", "wb"))
    logging.info(" [main] # finished!")

    logging.info(" [main] Training ...")
    dwe.train(dataloader, iter=args.train_iter, alpha=args.train_alpha, rate=1.0, ckpt_span=args.ckpt_span)
    logging.info(" [main] # finished!")

    logging.info(" [main] Save model ...")
    pickle.dump(dwe, open(f"../dwe_ckpt-{args.train_iter}.pkl", "wb"))
    logging.info(" [main] # finished!")


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--vocab", help="path of vocab")
    parser.add_argument("--time_start", type=int)
    parser.add_argument("--time_end", type=int)
    parser.add_argument("--time_span", type=int, default=1)
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument(
        "--diffusion",
        type=float,
        default=1,
        help="diffusion constant for variance in each time",
    )
    parser.add_argument("--positive", help="path of positive samples")
    parser.add_argument("--negative", help="path of negative samples")
    parser.add_argument(
        "--pretrain_iter", type=int, default=5000, help="number of pretrain steps"
    )
    parser.add_argument(
        "--pretrain_alpha",
        type=float,
        default=0.01,
        help="learning rate during pretrain",
    )
    parser.add_argument(
        "--train_iter", type=int, default=1000, help="number of train steps"
    )
    parser.add_argument(
        "--train_alpha", type=float, default=0.001, help="learning rate during training"
    )
    parser.add_argument(
        "--ckpt_span", type=int, default=10, help="(only fullbatch) save ckpt in each ckpt_span iter"
    )
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
