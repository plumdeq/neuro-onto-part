#!/usr/bin/env python3
# coding:utf8
"""
copyright Asan AGIBETOV <asan.agibetov@gmail.com>

Read in starspace embedding file -> compute clusters (with n_clusters given)
for all `__label__XXXXXX` (concept ids)

"""
# Standard-library imports
import logging
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
            if line[0].startswith('__label__'):
                E[line[0]] = np.array(list(map(float, line[1:]))).astype(np.float32)

    return E


@click.command()
@click.argument("emb-file", type=click.Path(exists=True))
@click.option("--output", default="sample_output.txt")
@click.option("--n-clusters", default=2)
def main(emb_file, output, n_clusters):
    logger.info("reading embeddings file {}".format(emb_file))
    my_embeddings = get_embeddings(emb_file)
    N, d = len(my_embeddings), len(list(my_embeddings.values())[0])
    X = np.zeros((N, d))
    labels = []
    for i, (label, emb) in enumerate(my_embeddings.items()):
        X[i] = emb
        labels.append(label)

    logger.info(X.shape)

    logger.info("training KMeans")
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)

    logger.info("output file to {}".format(output))
    with open(output, "w") as f:
        for label, emb in zip(labels, kmeans.labels_):
            f.write("{}:{}\n".format(label, emb))

if __name__ == "__main__":
    main()
