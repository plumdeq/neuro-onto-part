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


def read_logmap(logmap_f):
    lines = None

    logger.info("Reading in file {}".format(logmap_f))
    with open(logmap_f, "r") as f:
        lines = [l.strip() for l in f.readlines()]
        logger.info("Splitting on |")
        lines = [l.split("|") for l in lines]         

        # logger.info("Here is how I split them")
        # logger.info(lines[:5])

        logger.info("Splitting words and concepts on ;")
        # logger.info("Here is how I split them")
        lines = [(words.split(";"), ['__label__' + c for c in concepts.split(";")])
                 for (words, concepts) in lines]
        # logger.info(lines[:5])

    return lines


@time_me
def logmap_2_starspace(logmap_f, output):
    lines = read_logmap(logmap_f)
    logger.info("Writing starspace file to {}".format(output))
    with open(output, "w") as f:
        def write_out(words, concepts):
            fmt_str = " ".join(words)
            fmt_str = fmt_str + "\t"
            fmt_str = fmt_str + "\t".join(concepts)
            fmt_str = fmt_str + "\n"

            return fmt_str

        lines = [write_out(words, concepts) for words, concepts in lines]

        # logger.info("Here is the final representation")
        # logger.info(lines[:5])

        f.writelines(lines)


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


def get_unique_words(sets_of_words):
    words = fun.reduce(lambda x, y: x + y, sets_of_words)

    return list(set(words))


def check_missing_embeddings(words, E):
    """Check if there are any missing embeddings"""
    missing = [word for word in words if word not in E]
    logger.info("Missing embeddings for {} (sample)".format(missing[:10]))

    logger.info("Ratio of missing embeddings {}".format(len(missing)/len(words)))

    return missing


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
def make_starspace_embs(input_f, output_dir):
    logger.info("Computing starspace embeddings on {}".format(input_f))
    cmd_string = "-model {}/model -trainFile {}".format(output_dir, input_f)
    cmd_arg_list = " ".join([STARSPACE, DEFAULT_OPTS, cmd_string])

    logger.info("About to execute {}".format(cmd_arg_list))

    cmd_output = subprocess.run(shlex.split(cmd_arg_list), stdout=subprocess.PIPE)

    logger.info(cmd_output.stdout.decode("utf-8"))


    return None


@time_me
def do_cluster(aggregated_embs, sets_of_words, output_dir, prefix, cluster_size):
    logger.info("Computing KMeans for {} clusters".format(cluster_size))
    kmeans = kmeans_cluster(aggregated_embs, cluster_size)

    output_f = "-".join([os.path.join(output_dir, prefix), str(cluster_size)])
    logger.info("Writing clusters to {}".format(output_f))
    write_clusters(output_f, sets_of_words, kmeans)


def do_clusters(emb_f, logmap_f, output_dir, prefix):
    E = get_embeddings(emb_f)
    sets_of_words = get_sets_of_words(logmap_f)
    words = get_unique_words(sets_of_words)

    missing = check_missing_embeddings(words, E)

    logger.info("Computing aggregated embeddings per set of word (ignoring missing)")
    aggregated_embs = get_aggregated_embeddings(
            E, [word for word in sets_of_words if word not in missing])

    cluster_sizes = [2, 5, 10, 20, 50, 100, 200]
    # cluster_sizes = [20]
    cluster_times = []
    for cluster_size in cluster_sizes:
        _, time_do_cluster = do_cluster(
                aggregated_embs, sets_of_words, output_dir, prefix, cluster_size)
        fmt_string = "Kmeans with cluster size {} in {} seconds".format(
            cluster_size, time_do_cluster)
        cluster_times.append(fmt_string)
        logger.info(fmt_string)

    return cluster_times


def do_all(logmap_dir, converted_dir, output_dir, log_dir):
    _, _, logmap_files = next(os.walk(logmap_dir))
    logmap_files = [os.path.basename(f) for f in logmap_files]

    for f in logmap_files:
        logger.info("Doing {}".format(f))

        converted_dir_i = os.path.join(converted_dir, f)

        if not os.path.exists(converted_dir_i):
            logger.info("Creating {}".format(converted_dir_i))
            os.makedirs(converted_dir_i)

        logger.info("All converted files and intermediary steps in {}".format(
            converted_dir_i))

        logmap_f = os.path.join(logmap_dir, f)
        starspace_f = os.path.join(converted_dir_i, f + ".starspace")
        _, time_convert_to_starspace = logmap_2_starspace(logmap_f, starspace_f)

        _, time_starspace = make_starspace_embs(starspace_f, converted_dir_i)
        logger.info("Trained starspace model in {:.3f} seconds".format(
            time_starspace))

        output_dir_i = os.path.join(output_dir, f)

        if not os.path.exists(output_dir_i):
            logger.info("Creating {}".format(output_dir_i))
            os.makedirs(output_dir_i)

        logger.info("Results are written in {}".format(output_dir_i))
        starspace_embeddings_f = os.path.join(converted_dir_i, "model.tsv")
        cluster_times = do_clusters(starspace_embeddings_f, logmap_f, output_dir_i, "cluster")
        
        log_f = os.path.join(log_dir, f)
        logger.info("Logging into {}".format(log_f))

        if not os.path.exists(log_dir):
            logger.info("Creating {}".format(log_dir))
            os.makedirs(log_dir)

        with open(log_f, "w") as f:
            f.write("Time to convert to starspace input file {} seconds\n".format(
                time_convert_to_starspace))
            f.write("Starspace was trained with `{}`\n".format(DEFAULT_OPTS))
            f.write("Time to train embeddings with starspace {} seconds\n".format(
                time_starspace))
            f.write("Training clusters per cluster size\n")
            f.write("\n".join(cluster_times))


@click.command()
@click.argument("logmap_dir", type=click.Path(exists=True))
@click.option("--converted-dir", default="./converted")
@click.option("--results-dir", default="./results")
@click.option("--log-dir", default="./log")
def main(logmap_dir, converted_dir, results_dir, log_dir):
    do_all(logmap_dir, converted_dir, results_dir, log_dir)


if __name__ == "__main__":
    main()
