#!/usr/bin/env python3
# coding:utf8
"""
copyright Asan AGIBETOV <asan.agibetov@gmail.com>

Read in starspace embedding file -> compute clusters (with n_clusters given)
for sets of words 

"""
# Standard-library imports
import logging
import functools as fun
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Third-party imports
import click
import numpy as np
from sklearn.cluster import KMeans


def get_embeddings(emb_file):
    # build dictionary "idx: x1 x2"
    E = {}

    logger.info("Reading embeddings from {}".format(emb_file))
    with open(emb_file, "r") as f:
        for l in f.readlines():
            line = l.strip().split() 
            E[line[0]] = np.array(list(map(float, line[1:]))).astype(np.float32)

    return E


def get_sets_of_words(logmap_f):
    lines = None

    logger.info("Reading in file {}".format(logmap_f))
    with open(logmap_f, "r") as f:
        lines = [l.strip() for l in f.readlines()]
        logger.info("Splitting on |")
        lines = [l.split("|") for l in lines]         

        # logger.info("Here is how I split them")
        # logger.info(lines[:2])

        logger.info("Splitting words and concepts on ;")
        # logger.info("Here is how I split them")

        words = [words.split(";") for (words, concepts) in lines]
        # logger.info(words[:2])

        return words


def get_aggregated_embeddings(E, sets_of_words):
    """For each set of words compute aggregated vector representation"""
    embs = []

    for set_of_words in sets_of_words:
        existing_embeddings = [E[word] for word in set_of_words if word in E]

        if len(existing_embeddings) == 0:
            continue

        aggregated = fun.reduce(lambda x, y: x + y, existing_embeddings)
        embs.append(aggregated/len(set_of_words))

    N, d = len(embs), len(embs[0])
    X = np.zeros((N, d))
    for i in range(N):
        X[i] = embs[i]

    return X


@click.command()
@click.argument("logmap-f", type=click.Path(exists=True))
@click.argument("emb-file", type=click.Path(exists=True))
@click.option("--output", default="sample_output.txt")
@click.option("--n-clusters", default=2)
def main(logmap_f, emb_file, output, n_clusters):
    logger.info("reading embeddings file {}".format(emb_file))
    my_embeddings = get_embeddings(emb_file)
    sets_of_words = get_sets_of_words(logmap_f)

    N, d = len(my_embeddings), len(list(my_embeddings.values())[0])
    X = np.zeros((N, d))
    labels = []
    for i, (label, emb) in enumerate(my_embeddings.items()):
        X[i] = emb
        labels.append(label)

    logger.info(X.shape)

    logger.info("training KMeans")
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)


    with open(output, "w") as f:
        logger.info("Writing clusters to {}".format(output))
        for set_of_words, cluster in zip(sets_of_words, kmeans.labels_):
            words_str = ",".join(set_of_words)
            f.write("{}:{}\n".format(words_str, str(cluster)))


if __name__ == "__main__":
    main()
