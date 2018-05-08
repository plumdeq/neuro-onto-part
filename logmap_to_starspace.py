# coding:utf8
"""
copyright Asan AGIBETOV <asan.agibetov@gmail.com>

Simple script to convert LogMap stem -> concept_index mappings of the form

```
stem_1;stem_2|concept_index1,concept_index2
```

into the form which is required by the starspace toolkit

``` 
word1 word2 __label__concept_index1 __label__concept_index2
```

So we assume that we have mappings (stem -> word) and (concept_index -> __label__concept_index)

"""
# Standard-library imports
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Third-party imports
import click


@click.command()
@click.argument("logmap_f", type=click.Path(exists=True))
@click.option("--output", default="output.starspace")
def main(logmap_f, output):
    lines = None

    logger.info("Reading in file {}".format(logmap_f))
    with open(logmap_f, "r") as f:
        lines = [l.strip() for l in f.readlines()]
        logger.info("Splitting on |")
        lines = [l.split("|") for l in lines]         

        logger.info("Here is how I split them")
        logger.info(lines[:5])

        logger.info("Splitting words and concepts on ;")
        logger.info("Here is how I split them")
        lines = [(words.split(";"), ['__label__' + c for c in concepts.split(";")])
                 for (words, concepts) in lines]
        logger.info(lines[:5])


    logger.info("Writing starspace file to {}".format(output))
    with open(output, "w") as f:
        def write_out(words, concepts):
            fmt_str = " ".join(words)
            fmt_str = fmt_str + "\t"
            fmt_str = fmt_str + "\t".join(concepts)
            fmt_str = fmt_str + "\n"

            return fmt_str

        lines = [write_out(words, concepts) for words, concepts in lines]
        logger.info("Here is the final representation")
        logger.info(lines[:5])
        f.writelines(lines)


if __name__ == "__main__":
    main()
