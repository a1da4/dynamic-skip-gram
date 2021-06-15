import argparse
import logging

import numpy as np

from model import *
from ioutils import *


def main(args):
    logging.basicConfig(filename="train.log", filemode="w", level=logging.INFO)
    #logging.basicConfig(filename="train_debug.log", filemode="w", level=logging.DEBUG)

    logging.info(" [main] Initialization ...")
    vocab = load_vocab(args.vocab)
    dwe = SkipGramSmoothing(
        vocab=vocab,
        T=args.num_time_step,
        dim=args.dim,
        D=args.diffusion,
        taus=args.time_stamps,
        positive=args.positive,
        negative=args.negative
    )
    logging.debug(" [main] # model: SkipGramSmoothing")
    logging.debug(f" [main] # dwe.vocab: {len(dwe.vocab)} words")
    logging.debug(f" [main] # dwe.T: {dwe.T}")
    logging.debug(f" [main] # dwe.D: {dwe.D}")
    logging.debug(f" [main] # dwe.mean_target: {dwe.mean_target.shape}")
    logging.debug(f" [main] # dwe.mean_context: {dwe.mean_context.shape}")
    logging.debug(f" [main] # dwe.vals: {dwe.vals}")
    logging.debug(f" [main] # dwe.cov: \n{dwe.cov}")
    logging.debug(f" [main] # dwe.v (size): {dwe.v_target.shape}")
    logging.debug(f" [main] # dwe.v (value): \n{dwe.v_target[0][0]}")
    logging.debug(f" [main] # dwe.w (size): {dwe.w_target.shape}")
    logging.debug(f" [main] # dwe.w (value): \n{dwe.w_target[0][0]}")


    logging.info(" [main] Pre-training ...")
    dwe.train(iter=2, rate=0.1)
    logging.info(" [main] Training ...")
    dwe.train(iter=2, rate=1.0)


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", help="path of vocab")
    parser.add_argument("--num_time_step", type=int)
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument(
        "--diffusion",
        type=float,
        default=1,
        help="diffusion constant for variance in each time",
    )
    parser.add_argument("--time_stamps", type=int, nargs="*")
    parser.add_argument("--positive", help="path of positive samples")
    parser.add_argument("--negative", help="path of negative samples")
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
