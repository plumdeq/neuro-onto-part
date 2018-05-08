# Partitioning ontology alignment task with neural embeddings

This repo contains two scripts that prepare clusters of ontology classes from
the lexical index, which is computed with LogMap. The clusters themselves are
computed from the neural embeddings computed with StarSpace. Therefore it is
required that you have starspace installed somewhere on your system.

## Convert LogMap index into StarSpace-compliant format

First, you need to convert LogMap stem -> concept_index mappings of the form

```
stem_1;stem_2|concept_index1,concept_index2
```

into the form which is required by the starspace toolkit

``` 
word1 word2 __label__concept_index1 __label__concept_index2
```

So we assume that we have mappings (stem -> word) and (concept_index -> __label__concept_index).

To convert your `LOGMAP_INDEX` call the script like so

```
python3 logmap_to_starspace.py LOGMAP_INDEX

# see `python3 logmap_to_starspace --help` for help
```

## Computing clusters

`compute_clusters.py` takes in the converted lexical index files found in
`LOGMAP_DIR`, **and per each lexical index file it** computes embeddings with
starspace: `word1 x1 x2 ... xn`. Then, it computes aggregated vectors, and
finally outputs clusters per set of words. This script will populate: 

* `CONVERTED_DIR` with starspace model and embeddings 
* `RESULTS_DIR` with the clusters 
* `LOG_DIR` running time and other logging info

see `python3 compute_clusters.py --help` for more.

**Set the variable** `STARSPACE` **inside** `compute_clusters.py` **to point to the path to
your local binary of starspace**.
