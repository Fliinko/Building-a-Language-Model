# Building a Language Model

The Below is all built and run on the Python 3.8.5 Kernel.

        pip install requirements.txt

> This project builds a sentence generator given a corpus.

The corpora can be found in their respective folder 'Corpus'. Just a brief explanantion regarding the corpora, the corpora are all in Maltese and were provided through University of Malta resources.

The project generates a sentence given a pre-defined starting phrase from the user such as 
"Ilbierah kont" and the script attempts to build a sentence off of that phrase. Structurally, the generator works in an n-gram fashion but the main structures used to generate the sentences were the unigram, bigram and trigram. The perplexity for each n-gram model was also calculated
