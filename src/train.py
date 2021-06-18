import argparse
import logging
import pickle

import numpy as np

from ioutils import *
from model import *


def main(args):
    # logging.basicConfig(filename="train.log", filemode="w", level=logging.INFO)
    # logging.basicConfig(filename="train_debug.log", filemode="w", level=logging.DEBUG)
    logging.basicConfig(level=logging.DEBUG)

    logging.info(" [main] Initialization ...")
    vocab = load_vocab(args.vocab)
    time_range = args.time_end - args.time_start + 1
    dwe = SkipGramSmoothing(
        vocab=vocab,
        T=time_range,
        dim=args.dim,
        D=args.diffusion,
        taus=range(args.time_start, args.time_end + 1),
        positive=args.positive,
        negative=args.negative,
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
    logging.debug(f" [main] # dwe.v (value): \n{dwe.v_target[0][0]}")
    logging.debug(f" [main] # dwe.w (size): {dwe.w_target.shape}")
    logging.debug(f" [main] # dwe.w (value): \n{dwe.w_target[0][0]}")

    logging.info(" [main] Pre-training ...")
    dwe.train(iter=args.pretrain_iter, rate=0.1)
    logging.info(" [main] # finished!")

    logging.info(" [main] Training ...")
    dwe.train(iter=args.train_iter, rate=1.0)
    logging.info(" [main] # finished!")

    logging.info(" [main] Save model ...")
    pickle.dump(dwe, open("../dwe.pkl", "wb"))
    logging.info(" [main] # finished!")
    exit()


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", help="path of vocab")
    parser.add_argument("--time_start", type=int)
    parser.add_argument("--time_end", type=int)
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument(
        "--diffusion",
        type=float,
        default=1,
        help="diffusion constant for variance in each time",
    )
    parser.add_argument("--positive", help="path of positive samples")
    parser.add_argument("--negative", help="path of negative samples")
    parser.add_argument("--pretrain_iter", type=int, default=5000, help="iteration")
    parser.add_argument("--train_iter", type=int, default=1000, help="iteration")
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
