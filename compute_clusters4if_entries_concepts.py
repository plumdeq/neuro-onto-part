# coding:utf8
"""
copyright Asan AGIBETOV <asan.agibetov@gmail.com>

Takes on input embeddings: 
word1 x1 x2

* compute aggregated vectors
* output clusters per set of words

"""
# Standard-library imports
import time
import functools as fun
import logging
import os
import shlex
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Third-party imports
import click
import numpy as np
from sklearn.cluster import KMeans


STARSPACE = "/home/asan/code/kg-dl/vendor-code/starspace/starspace"
DEFAULT_OPTS = 'train -trainMode 0 -similarity dot -label __label__ --epoch 100 --dim 64'


def time_me(f):
    def another_f(*args, **kwargs):
        start = time.time()
        res = f(*args, **kwargs)
        delta = time.time() - start

        return res, delta

    return another_f





def get_embeddings(emb_file):
    # build dictionary "idx: x1 x2"
    E = {}

    logger.info("Reading embeddings from {}".format(emb_file))
    with open(emb_file, "r") as f:
        for l in f.readlines():
            line = l.strip().split() 
            E[line[0]] = np.array(list(map(float, line[1:]))).astype(np.float32)

    return E



def get_embeddings_concepts(emb_file):
    # build dictionary "idx: x1 x2"
    E = {}

    logger.info("Reading embeddings from {}".format(emb_file))
    with open(emb_file, "r") as f:
        for l in f.readlines():
            line = l.strip().split()
            if line[0].startswith('__label__'):
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

        words_list = [words.split(";") for (words, concepts) in lines]
        # logger.info(words[:2])


        concepts_list = [concepts.split(";") for (words, concepts) in lines]


        return words_list


def get_sets_of_concepts(logmap_f):
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

        words_list = [words.split(";") for (words, concepts) in lines]
        # logger.info(words[:2])

        concepts_list = [concepts.split(";") for (words, concepts) in lines]


        return concepts_list




def get_unique_words(sets_of_words):
    words = fun.reduce(lambda x, y: x + y, sets_of_words)

    return list(set(words))


def check_missing_embeddings(words, E):
    """Check if there are any missing embeddings"""
    missing = [word for word in words if word not in E]
    logger.info("Missing embeddings for {} (sample)".format(missing[:10]))

    logger.info("Ratio of missing embeddings {}".format(len(missing)/len(words)))

    return missing


def get_aggregated_embeddings_words(E, sets_of_words):
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




#def getEmbedding4Concept(E, conceptid):
#    return E["__label__"conceptid]
    



def get_aggregated_embeddings_concepts(E, sets_of_concepts):
    """For each set of concept compute aggregated vector representation"""
    embs = []

    for set_of_concepts in sets_of_concepts:
        #existing_embeddings = [E[concept] for concept in set_of_concepts if concept in E]
        existing_embeddings = [E["__label__"+concept] for concept in set_of_concepts]

        if len(existing_embeddings) == 0:
            continue

        aggregated = fun.reduce(lambda x, y: x + y, existing_embeddings)
        embs.append(aggregated/len(set_of_concepts))

    N, d = len(embs), len(embs[0])
    X = np.zeros((N, d))
    for i in range(N):
        X[i] = embs[i]

    return X



def concatenate_embs(aggregated_embs_words, aggregated_embs_concepts):
    
    #print(len(aggregated_embs_words[0]))

    N, d = len(aggregated_embs_words), len(aggregated_embs_words[0])+len(aggregated_embs_concepts[0])
    X = np.zeros((N, d))
    for i in range(N):
        X[i] = np.concatenate((aggregated_embs_words[i], aggregated_embs_concepts[i]))

    return X




def kmeans_cluster(aggregated_embs, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(aggregated_embs)

    return kmeans


def write_clusters(output_f, sets_of_words, kmeans):
    with open(output_f, "w") as f:
        logger.info("Writing clusters to {}".format(output_f))
        for set_of_words, cluster in zip(sets_of_words, kmeans.labels_):
            words_str = ",".join(set_of_words)
            f.write("{}:{}\n".format(words_str, str(cluster)))




@time_me
def do_cluster(aggregated_embs, sets_of_words, output_f, cluster_size):
    logger.info("Computing KMeans for {} clusters".format(cluster_size))
    kmeans = kmeans_cluster(aggregated_embs, cluster_size)

    #output_f = "-".join([os.path.join(output_dir, prefix), str(cluster_size)])
    logger.info("Writing clusters to {}".format(output_f))
    write_clusters(output_f, sets_of_words, kmeans)


def do_clusters(emb_f, logmap_f, output_file, cluster_size):
    E = get_embeddings(emb_f)
    F = get_embeddings_concepts(emb_f)
    
    sets_of_words = get_sets_of_words(logmap_f)
    sets_of_concepts = get_sets_of_concepts(logmap_f)
    
    words = get_unique_words(sets_of_words)

    missing = check_missing_embeddings(words, E)

    logger.info("Computing aggregated embeddings per set of word (ignoring missing)")
    aggregated_embs_words = get_aggregated_embeddings_words(
            E, [word for word in sets_of_words if word not in missing])


    #We use only the concept embeddings
    aggregated_embs_concepts = get_aggregated_embeddings_concepts(
            F, sets_of_concepts)


    ##Aggregate or concatenate embeddings. We have on one had a vector for teh aggregated word embeddings and then another vector for the aggregated concept embeddings
    ##for each if entry: word1;word2|concept1;concept2
    #aggregated_embs = concatenate_embs(aggregated_embs_words, aggregated_embs_concepts)


    #cluster_sizes = [2, 5, 10, 20, 50, 100, 200]
    # cluster_sizes = [20]
    cluster_times = []
    #for cluster_size in cluster_sizes:
    
    _, time_do_cluster = do_cluster(
                aggregated_embs_concepts, sets_of_words, output_file, cluster_size)  
    fmt_string = "Kmeans with cluster size {} in {} seconds".format(
            cluster_size, time_do_cluster)
    cluster_times.append(fmt_string)
    logger.info(fmt_string)

    return cluster_times




@click.command()
@click.argument("logmap-f", type=click.Path(exists=True))
@click.argument("emb-file", type=click.Path(exists=True))
@click.option("--output", default="sample_output.txt")
@click.option("--n-clusters", default=2)
def main(logmap_f, emb_file, output, n_clusters):
    cluster_times = do_clusters(emb_file, logmap_f, output, n_clusters)


if __name__ == "__main__":
    main()
