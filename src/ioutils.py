# 主に何度も使いそうな関数（vocab などの入出力）
import logging
import os


def save_vocab(vocab):
    logging.info(" [save_vocab] saving vocaburaly...")
    with open("../vocab.txt", "w") as fp:
        for word in vocab:
            fp.write(f"{word}\n")
    logging.info(" [save_vocab] finished!")


def load_vocab(vocab_path):
    logging.info(" [load_vocab] loading vocaburaly...")
    vocab = []
    with open(vocab_path) as fp:
        for line in fp:
            word = line.strip()
            vocab.append(word)
    logging.info(" [load_vocab] finished!")
    return vocab


def save_2d_matrix(matrix, name):
    logging.info(" [save_2d_matrix] saving matrix...")
    with open(f"../{name}.txt", "w") as fp:
        for col in matrix:
            col_nonzero = [f"{id}:{v}" for id, v in enumerate(col) if v > 0]
            fp.write(f"{' '.join(col_nonzero)}\n")
    logging.info(" [save_2d_matrix] finished!")


def save_3d_matrix(matrix, name):
    logging.info(" [save_3d_matrix] saving matrix...")
    os.makedirs(f"../{name}", exist_ok=True)
    for i, matrix_2d in enumerate(matrix):
        save_2d_matrix(matrix_2d, f"{name}/{i}")
    logging.info(" [save_3d_matrix] finished!")

def load_2d_matrix(matrix_path, x_size, y_size):
    logging.info(" [load_2d_matrix] loading matrix...")
    matrix_2d = [[0 for _ in range(y_size)] for _ in range(x_size)]
    with open(matrix_path) as fp:
        x_id = 0
        for line in fp:
            y_id_vals = line.strip().split()
            for y_id_val in y_id_vals:
                y_id, val = y_id_val.split(":")
                matrix_2d[x_id][int(y_id)] += float(val)
    logging.info(" [load_2d_matrix] finished!")
    return matrix_2d

def load_3d_matrix(matrix_path, z_size, x_size, y_size):
    logging.info(" [load_3d_matrix] loading matrix...")
    matrix_3d = []
    for i in range(z_size):
        matrix_2d = load_2d_matrix(f"{matrix_path}/{i}.txt", x_size, y_size)
        matrix_3d.append(matrix_2d)
        del matrix_2d
    logging.info(" [load_3d_matrix] finished!")
    return matrix_3d
