In `/text_relationship` fold, first create a fold to store text embedding:

```shell
mkdir pre-trained
cd pre-trained
```

The text embedding we will use GloVe, see [github](https://github.com/stanfordnlp/GloVe), and [project page](https://nlp.stanford.edu/data/glove.840B.300d.zip).

download pre-trained word vectors, I choose Common Crawl, which should exist 2.03 GB space free.

```shell
weget https://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
cd ..
```

Then, run demo to get the relationship matrix:

```shell
python main.py --split <which split> --method <which method>
```


