import argparse
import logging
import pickle

import numpy as np
import matplotlib.pyplot as plt

from functions import cos
from model import *


def most_similar(dwe, target_vec_t, t, topk):
    sims = np.zeros(dwe.V)
    for i, context_word in enumerate(dwe.vocab):
        context_vecs = dwe.predict(context_word)
        context_vec_t = context_vecs[t]
        sims[i] += cos(target_vec_t, context_vec_t)
    return np.argsort(-1*sims)[1:topk+1]

def main(args):
    # logging.basicConfig(filename="../visualize.log", filemode="w", level=logging.INFO)
    logging.basicConfig(filename="../visualize_debug.log", filemode="w", level=logging.DEBUG)

    logging.debug(" [main] Load model ...")
    dwe = pickle.load(open(args.model, "rb"))
    logging.debug(f" [main] # dwe.vocab: {len(dwe.vocab)} words")
    logging.debug(f" [main] # dwe.T: {dwe.T}")
    logging.debug(f" [main] # dwe.D: {dwe.D}")
    logging.debug(f" [main] # dwe.mean_target: {dwe.mean_target.shape}")
    logging.debug(f" [main] # dwe.mean_context: {dwe.mean_context.shape}")
    logging.debug(f" [main] # dwe.vals: {dwe.vals}")
    logging.debug(f" [main] # dwe.precision: \n{dwe.precision}")
    logging.debug(f" [main] # dwe.v (size): {dwe.v_target.shape}")
    logging.debug(f" [main] # dwe.w (size): {dwe.w_target.shape}")
    v_nan = 0
    w_nan = 0
    for v in range(dwe.V):
        for d in range(dwe.D):
            v_nan += np.any(np.isnan(dwe.v_target[v][d]))
            w_nan += np.any(np.isnan(dwe.w_target[v][d]))
    logging.debug(f" [main] # dwe.v % isnan: {v_nan / (dwe.V*dwe.D)}")
    logging.debug(f" [main] # dwe.w % isnan: {w_nan / (dwe.V*dwe.D)}")


    # TODO target, context words を与えられるように
    logging.debug(" [main] Obtain vectors ...")
    target_word = "gay"
    target_word = "peer"
    logging.debug(f" [main] # word: {target_word}")
    target_vecs = dwe.predict(target_word)
    logging.debug(f" [main] # vectors ({dwe.T}, {dwe.D}): \n{target_vecs}")

    """
    closest_begin = most_similar(dwe, target_vecs[0], 0, topk=5)
    logging.debug(f" [main] # closest (begin): {[dwe.vocab[id] for id in closest_begin]}")
    closest_end = most_similar(dwe, target_vecs[-1], dwe.T-1, topk=5)
    logging.debug(f" [main] # closest (end): {[dwe.vocab[id] for id in closest_end]}")

    context_words = []
    for id in closest_begin:
        context_words.append(dwe.vocab[id])
    for id in closest_end:
        context_words.append(dwe.vocab[id])
    """

    context_words = [
        "happy", 
        "bright",
        "cheerful",
        "pleasant",
        "witty",
        "marry",
        "sex",
    ]

    context_words = [
        "classroom",
        "",
        "networks",
        "teacher",
        "parent",
        "noble",
        "lawyer",
        "baron",
        "member",
        "lord",
    ]

    plt.xlabel("date")
    plt.ylabel("cosine distance")
    for context_word in context_words:
        if context_word in dwe.vocab:
            logging.debug(f" [main] # word: {context_word}")
            context_vecs = dwe.predict(context_word)
            distances = []
            for t in range(dwe.T):
                # 絶対値を引いている？
                #distance = 1 - np.abs(cos(target_vecs[t], context_vecs[t]))
                distance = (1 - cos(target_vecs[t], context_vecs[t])) / 2
                logging.debug(f" [main] ## time-{dwe.time_stamps[t]}, cos dist: {distance}")
                distances.append(distance)
            color = "blue" if context_words.index(context_word) < 5 else "red"
            plt.plot(dwe.time_stamps, distances, label=f"{context_word}", color=color)
        else:
            logging.debug(f" [main] # word: {context_word} not in vocab")
    plt.legend()
    plt.savefig(f"../{target_word}.png")


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="path to saved model.pkl")
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
